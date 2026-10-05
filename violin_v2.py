"""
violin_v2.py
============
Text-level significance testing + violin plot of CAROB (conjoint adjacency
ratio over baseline = bifolio adjacency rate / random baseline), working
directly off `results_full_all.csv` as written by sweep_v2.py.

The analysis is the same one-point-per-source-text design as violin.py: every
row for a given (quire construction, source text) -- all windows, words-per-
folio values, distance metrics and SA restarts -- is first collapsed to a
single MEDIAN CAROB, so the statistical unit is the DOCUMENT, not the run. (The
10 bootstrap windows per text in sweep_v2 are replicates within a document and
are collapsed along with everything else.)

Differences from violin.py
--------------------------
* Input is sweep_v2's results_full_all.csv (not an xlsx); the value is the
  `carob` column; groups come from `bifolios_per_quire` (1-5 ->
  Singulion ... Quinion).
* Optional filters before aggregation: --wpf, --metrics, --best-cost-only
  (keep only each cell's lowest-seriation-cost restart, which is chosen
  without looking at ground truth). Default: no filtering, as in violin.py.
  --metrics and --best-cost-only apply to baseline and VMS rows alike; --wpf
  applies only to rows that have a words_per_folio value, so VMS rows run with the
  real foliation (no such column / empty value) are exempt rather than dropped.
  If filters do remove every VMS row, a warning is printed.
* The y-axis limit is automatic and always includes the VMS value (violin.py
  hard-coded 20, which could clip VMS off the plot without any sign of it). With
  --ymax below the VMS value, VMS is marked at the top edge as "off scale" and a
  warning is printed. A reference line at CAROB = 1 (random ordering) is drawn,
  and the legend sits above the axes so it never covers data.
* VMS rows are recognised by source_text matching --vms-pattern, or can be
  supplied as a separate CSV with the same schema via --vms-input. All VMS rows
  collapse to ONE value (N = 1). This assumes VMS was run with its real
  foliation, so that bifolios_per_quire is not a meaningful factor for it; a
  warning is printed if the VMS rows span several bifolios_per_quire values.
* Seeds are command-line controlled; the settings used are written to
  run_settings.json.
* Statements in violin.py's docstring about how specific distribution families
  fit the earlier dataset are NOT carried over: with new data the family is
  selected afresh by AIC, and distribution_fit_comparison.csv shows the result.

WHAT IT PRODUCES  (in --output-dir, default ./violin_v2_out)
----------------
1. text_level_values.csv
       One row per (group, source_text): median CAROB used everywhere else,
       plus n_rows, n_windows and the 25th/75th percentiles for auditing.
       The VMS value (if any) is included as its own group.
2. pairwise_wilcoxon_text_level.csv
       Wilcoxon signed-rank test among quire-construction groups, Holm-
       corrected. Every text is seriated under every quire construction, so the
       groups are matched samples (repeated measures); the paired test on
       per-text differences is the appropriate one. Effect size is the
       matched-pairs rank-biserial correlation.
3. pairwise_bootstrap_text_level.csv
       Paired/cluster bootstrap (resample source texts, keeping each text's
       values in all groups together) of the median paired difference: an
       assumption-light cross-check on Wilcoxon.
4. friedman.txt
       Omnibus Friedman test across all groups (needs >= 3 groups).
5. vms_membership_report.csv  (only if VMS data is present)
       For each group, three complementary answers to "is VMS a plausible
       member of this group's distribution":
         a) EXACT RANK TEST: VMS + the n texts as n+1 exchangeable draws; exact,
            but its two-sided p cannot go below 2/(n+1).
         b) EMPIRICAL TAIL FRACTION: fraction of the n texts >= VMS (floors at 0).
         c) PARAMETRIC-BOOTSTRAP TAIL PROBABILITY: four candidate families
            (log-normal, gamma, Weibull, normal) are fitted to the group's
            text-level values and the best by AIC is used; P(new draw >= VMS)
            is computed under it, and sample-size uncertainty is propagated by
            refitting the SAME family on bootstrap resamples. This can express
            probabilities below 1/(n+1), at the price of assuming some
            parametric shape -- so every family's AIC is written to (6) for
            audit. Note this is an UPPER-tail probability.
6. distribution_fit_comparison.csv
       Per-group AIC (and optimistic KS diagnostics) for every candidate family.
7. violin_text_level.svg
       One point per source text per group. VMS (if present) is charted at its
       MEDIAN CAROB over all VMS runs: a dashed line,
       inside nested orange percentile bands of the VMS run values (default: central
       50% = IQR, darker; central 90% = 5th-95th percentile, paler; see --vms-bands).
       Bands need >= 5 VMS runs. An off-scale marker is drawn at the top edge if the
       median exceeds --ymax. NOTE the bands show run-to-run variability within the
       VMS (distance metrics x SA restarts), not between-text variability, so they are
       not directly comparable with the spread of the violins.
8b. vms_percentiles.csv
       The VMS summary behind the bands: n_runs, min, median, max, per-metric
       medians, and each band's lower/upper edge.
8. run_settings.json
       Input path, filters, row counts, seeds and package versions.

Usage
-----
    python violin_v2.py --input outputs_v2/results_full_all.csv
    python violin_v2.py --input outputs_v2/results_full_all.csv --vms-input vms_results.csv
    python violin_v2.py --input outputs_v2/results_full_all.csv --wpf 200 --metrics hellinger
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import stats

VALUE_COL = "carob"
REQUIRED_COLS = {"source_text", "bifolios_per_quire", VALUE_COL}
LABEL_MAP = {1: "Singulion", 2: "Binion", 3: "Ternion",
             4: "Quaternion", 5: "Quinion"}
VMS_LABEL = "VMS"
VMS_PATTERN_DEFAULT = r"voynich|vms"
N_BOOT_DEFAULT = 20000
VMS_BANDS_DEFAULT = [50.0, 90.0]   # central coverage (%) of the nested VMS bands
MIN_RUNS_FOR_BANDS = 5             # fewer VMS runs than this: percentiles are not meaningful
SEED_DEFAULT = 42
_DUP_KEY = ["source_text", "window_idx", "words_per_folio",
            "bifolios_per_quire", "metric", "restart"]


# =============================================================================
# LOAD + FILTER + AGGREGATE
# =============================================================================

def bpq_label(n: int) -> str:
    return LABEL_MAP.get(int(n), f"{int(n)} bifolios")


def read_results(path: Path) -> pd.DataFrame:
    """Read a sweep_v2-style results CSV and validate/clean the columns we need."""
    df = pd.read_csv(path)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required column(s) {sorted(missing)}. Expected "
            f"sweep_v2.py's results_full_all.csv. Columns found: {list(df.columns)}")
    df["source_text"] = df["source_text"].astype(str)
    df["bifolios_per_quire"] = pd.to_numeric(df["bifolios_per_quire"], errors="coerce")
    df[VALUE_COL] = pd.to_numeric(df[VALUE_COL], errors="coerce")
    bad = ~np.isfinite(df[VALUE_COL]) | df["bifolios_per_quire"].isna()
    if bad.any():
        print(f"  WARNING: dropping {int(bad.sum())} row(s) of {len(df)} in {path.name} with "
              f"non-finite CAROB (e.g. random baseline = 0) or missing bifolios_per_quire.")
        df = df.loc[~bad].copy()
    df["bifolios_per_quire"] = df["bifolios_per_quire"].astype(int)
    if "best_cost_restart" in df.columns:
        df["best_cost_restart"] = (df["best_cost_restart"].astype(str)
                                   .str.lower().isin(["true", "1"]))
    return df


def apply_filters(df: pd.DataFrame, wpf: list[int] | None, metrics: list[str] | None,
                  best_only: bool, exempt_missing_wpf: bool = False) -> pd.DataFrame:
    """Restrict rows before aggregation (no-op with default arguments).

    exempt_missing_wpf: if True, the --wpf filter leaves alone any row that has no
    words_per_folio value (or if the column is absent). Used for VMS rows run with
    the manuscript's real foliation, whose folio lengths are not a sweep parameter
    and which must not be dropped by a filter meant for the synthetic texts. VMS rows
    that DO carry a words_per_folio value (synthetic foliation) are still filtered.
    """
    needs = [(metrics, "metric"), (best_only, "best_cost_restart")]
    if not exempt_missing_wpf:
        needs.insert(0, (wpf, "words_per_folio"))
    for flag, col in needs:
        if flag and col not in df.columns:
            raise ValueError(f"Filter requested but column {col!r} is not in the input.")
    if wpf and "words_per_folio" in df.columns:
        keep = df["words_per_folio"].isin(wpf)
        if exempt_missing_wpf:
            keep = keep | df["words_per_folio"].isna()
        df = df[keep]
    if metrics:
        df = df[df["metric"].isin(metrics)]
    if best_only:
        df = df[df["best_cost_restart"]]
    return df.copy()


def split_baseline_vms(df: pd.DataFrame, pattern: str,
                       vms_extra: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Strip VMS-named rows out of the baseline; return (baseline, vms_or_None).
    Rows from a separate --vms-input file are always treated as VMS."""
    mask = df["source_text"].str.contains(pattern, case=False, regex=True)
    baseline = df.loc[~mask].copy()
    vms = df.loc[mask].copy()
    if vms_extra is not None:
        vms = pd.concat([vms, vms_extra], ignore_index=True)
    if vms.empty:
        return baseline, None
    vms["bpq_label"] = VMS_LABEL
    return baseline, vms


def check_duplicates(df: pd.DataFrame, label: str) -> None:
    if all(c in df.columns for c in _DUP_KEY):
        n = int(df.duplicated(_DUP_KEY).sum())
        if n:
            print(f"  WARNING: {n} duplicated (text, window, wpf, bpq, metric, restart) "
                  f"key(s) in {label}. Were results from several sweep runs mixed? "
                  f"They would be pooled into the same text-level median.")


def text_level_aggregate(df: pd.DataFrame, group_col: str = "bpq_label") -> pd.DataFrame:
    """Collapse every (window x wpf x metric x restart) row to a single median
    CAROB per (group, source_text) -- the true independent unit of replication."""
    keys = [group_col, "source_text"]
    g = df.groupby(keys)[VALUE_COL]
    out = g.agg(text_level_median="median", n_rows="size",
                q25=lambda s: s.quantile(0.25),
                q75=lambda s: s.quantile(0.75)).reset_index()
    if "window_idx" in df.columns:
        nw = df.groupby(keys)["window_idx"].nunique().rename("n_windows").reset_index()
        out = out.merge(nw, on=keys)
    return out


# =============================================================================
# PAIRWISE TESTS AMONG BASELINE GROUPS
# =============================================================================

def holm_adjust(p: np.ndarray) -> np.ndarray:
    """Holm step-down adjustment; NaN p-values are ignored (and stay NaN)."""
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    idx = np.where(~np.isnan(p))[0]
    if not len(idx):
        return out
    order = idx[np.argsort(p[idx])]
    m, running = len(order), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(p[i] * (m - rank), 1.0))
        out[i] = running
    return out


def build_paired_matrix(baseline_text: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    """
    Pivot text-level medians to a (source_text x group) matrix. Raises if any
    cell is missing: paired tests are only valid on a fully-crossed design, and
    a silent NaN-drop per comparison would change the effective N test-to-test.
    """
    pivot = baseline_text.pivot(index="source_text", columns="bpq_label",
                                values="text_level_median")
    missing = pivot[groups].isna().sum()
    if missing.any():
        raise ValueError(
            "build_paired_matrix: not every source_text has a value in every "
            f"group -- design is not fully crossed. Missing counts:\n{missing}")
    return pivot[groups]


def run_pairwise_wilcoxon(pivot: pd.DataFrame, groups: list[str],
                          alpha: float = 0.05) -> pd.DataFrame:
    """Holm-corrected pairwise Wilcoxon signed-rank tests on matched per-text
    values; effect size = matched-pairs rank-biserial r = (W+ - W-)/(W+ + W-)."""
    rows = []
    for a, b in itertools.combinations(groups, 2):
        xa, xb = pivot[a].values, pivot[b].values
        diffs = xa - xb
        try:
            stat, p = stats.wilcoxon(xa, xb, alternative="two-sided",
                                     zero_method="wilcox", method="exact")
        except ValueError:
            stat, p = np.nan, np.nan
        nz = diffs[diffs != 0]
        if len(nz):
            ranks = stats.rankdata(np.abs(nz))
            w_pos, w_neg = ranks[nz > 0].sum(), ranks[nz < 0].sum()
            r = (w_pos - w_neg) / (w_pos + w_neg)
        else:
            r = np.nan
        rows.append(dict(group_a=a, group_b=b, n_pairs=len(xa),
                         statistic=stat, p_raw=p,
                         effect_size_rank_biserial=r,
                         median_diff=float(np.median(diffs))))
    df = pd.DataFrame(rows)
    df["p_holm"] = holm_adjust(df["p_raw"].values)
    df["significant"] = df["p_holm"] < alpha
    return df.sort_values("p_raw").reset_index(drop=True)


def bootstrap_pairwise_median_diff(pivot: pd.DataFrame, groups: list[str],
                                   n_boot: int = N_BOOT_DEFAULT, seed: int = SEED_DEFAULT,
                                   alpha: float = 0.05) -> pd.DataFrame:
    """Paired/cluster bootstrap: resample SOURCE TEXTS (rows of `pivot`) with
    replacement so each resampled text's values in both groups stay together;
    bootstrap the median of the paired differences; percentile CI + two-sided
    bootstrap p-value (Holm-corrected)."""
    rng = np.random.default_rng(seed)
    n_texts = len(pivot)
    rows = []
    for a, b in itertools.combinations(groups, 2):
        diffs = pivot[a].values - pivot[b].values
        idx_boot = rng.integers(0, n_texts, size=(n_boot, n_texts))
        boot_diffs = np.median(diffs[idx_boot], axis=1)
        ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])
        p_boot = min(2 * min(np.mean(boot_diffs <= 0), np.mean(boot_diffs >= 0)), 1.0)
        rows.append(dict(group_a=a, group_b=b, n_pairs=n_texts,
                         observed_median_diff=float(np.median(diffs)),
                         boot_ci_lo=ci_lo, boot_ci_hi=ci_hi, p_bootstrap=p_boot))
    df = pd.DataFrame(rows)
    df["p_bootstrap_holm"] = holm_adjust(df["p_bootstrap"].values)
    df["significant"] = df["p_bootstrap_holm"] < alpha
    return df.sort_values("p_bootstrap").reset_index(drop=True)


# =============================================================================
# VMS MEMBERSHIP: exact rank test + empirical tail + parametric bootstrap
# =============================================================================

def exact_rank_test(vms_value: float, group_vals: np.ndarray) -> tuple[int, float]:
    """Exact two-sided rank test treating VMS + n texts as n+1 exchangeable
    draws. The smallest possible p is 2/(n+1)."""
    n = len(group_vals)
    rank = int((group_vals < vms_value).sum()) + 1
    pos_from_edge = min(rank, (n + 1) - rank + 1)
    return rank, min(2 * pos_from_edge / (n + 1), 1.0)


# All four candidates have 2 free parameters (floc=0 for the positive-support
# families), so AIC differences reduce to log-likelihood differences.
CANDIDATE_DISTS: dict[str, tuple] = {
    "lognorm": (stats.lognorm, dict(floc=0)),
    "gamma": (stats.gamma, dict(floc=0)),
    "weibull_min": (stats.weibull_min, dict(floc=0)),
    "norm": (stats.norm, dict()),
}


def fit_all_candidates(vals: np.ndarray) -> dict:
    """Fit every candidate family; families that fail or give a non-finite
    log-likelihood are skipped. KS p-values use parameters estimated from the
    same sample and are therefore optimistic -- a rough diagnostic only."""
    out = {}
    for name, (dist, kwargs) in CANDIDATE_DISTS.items():
        try:
            params = dist.fit(vals, **kwargs)
            loglik = float(np.sum(dist.logpdf(vals, *params)))
            if not np.isfinite(loglik):
                continue
            aic = 2 * 2 - 2 * loglik
            ks_stat, ks_p = stats.kstest(vals, name, args=params)
            out[name] = dict(params=params, loglik=loglik, aic=aic,
                             ks_stat=float(ks_stat), ks_p=float(ks_p))
        except Exception:
            continue
    if not out:
        raise RuntimeError("No candidate distribution could be fit to this group's values.")
    return out


def parametric_membership_probability(vms_value: float, group_vals: np.ndarray,
                                      n_boot: int = N_BOOT_DEFAULT,
                                      seed: int = SEED_DEFAULT) -> dict:
    """AIC-selected family's P(new draw >= VMS), with a bootstrap interval from
    refitting the SAME family (chosen once, from the full sample) on resamples
    of the n texts."""
    rng = np.random.default_rng(seed)
    n = len(group_vals)
    fits = fit_all_candidates(group_vals)
    best_name = min(fits, key=lambda k: fits[k]["aic"])
    dist, kwargs = CANDIDATE_DISTS[best_name]
    point = float(dist.sf(vms_value, *fits[best_name]["params"]))

    boot = np.empty(n_boot)
    for i in range(n_boot):
        resample = rng.choice(group_vals, size=n, replace=True)
        try:
            boot[i] = float(dist.sf(vms_value, *dist.fit(resample, **kwargs)))
        except Exception:
            boot[i] = np.nan
    boot = boot[~np.isnan(boot)]
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5]) if len(boot) else (np.nan, np.nan)
    return dict(
        distribution=best_name,
        aic_table={k: f["aic"] for k, f in fits.items()},
        ks_table={k: (f["ks_stat"], f["ks_p"]) for k, f in fits.items()},
        point_estimate=point,
        boot_median=float(np.median(boot)) if len(boot) else np.nan,
        boot_ci_lo=float(ci_lo), boot_ci_hi=float(ci_hi),
        n_valid_boot=len(boot),
    )


def vms_membership_report(vms_value: float, baseline_text: pd.DataFrame,
                          groups: list[str], n_boot: int,
                          seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(report_df, aic_comparison_df): one row per group / per (group, family)."""
    rows, aic_rows = [], []
    for g in groups:
        vals = baseline_text.loc[baseline_text["bpq_label"] == g, "text_level_median"].values
        n = len(vals)
        if n == 0:
            continue
        rank, p_exact = exact_rank_test(vms_value, vals)
        row = dict(group=g, n_texts=n,
                   group_median=np.median(vals), group_min=np.min(vals), group_max=np.max(vals),
                   vms_value=vms_value, vms_rank_among_n_plus_1=rank,
                   p_exact_rank_test=p_exact,
                   empirical_tail_fraction=float((vals >= vms_value).mean()))
        try:
            pr = parametric_membership_probability(vms_value, vals, n_boot=n_boot, seed=seed)
        except RuntimeError as e:
            print(f"  WARNING: {g}: {e} Parametric columns left empty.")
            pr = None
        if pr:
            row.update(best_fit_distribution=pr["distribution"],
                       parametric_tail_prob_point=pr["point_estimate"],
                       parametric_tail_prob_boot_median=pr["boot_median"],
                       parametric_tail_prob_ci_lo=pr["boot_ci_lo"],
                       parametric_tail_prob_ci_hi=pr["boot_ci_hi"])
            best = min(pr["aic_table"].values())
            for name, aic_val in pr["aic_table"].items():
                ks_stat, ks_p = pr["ks_table"][name]
                aic_rows.append(dict(group=g, distribution=name, aic=aic_val,
                                     delta_aic=aic_val - best, ks_stat=ks_stat, ks_p=ks_p,
                                     selected=(name == pr["distribution"])))
        rows.append(row)
    aic_df = pd.DataFrame(aic_rows)
    if not aic_df.empty:
        aic_df = aic_df.sort_values(["group", "aic"]).reset_index(drop=True)
    return pd.DataFrame(rows), aic_df


def vms_band_table(values: np.ndarray, coverages: list[float]) -> list[tuple[float, float, float]]:
    """Nested central percentile bands of the VMS run values: for a coverage c (%),
    the interval [50 - c/2, 50 + c/2] percentiles (e.g. 50 -> the interquartile
    range, 90 -> the 5th-95th percentile). Returned widest first."""
    out = []
    for c in sorted(set(coverages), reverse=True):
        lo, hi = np.percentile(values, [50 - c / 2, 50 + c / 2])
        out.append((float(c), float(lo), float(hi)))
    return out


# =============================================================================
# PLOTTING
# =============================================================================

def make_violin(text_data: dict[str, np.ndarray], groups: list[str],
                vms_value: float | None, out_path: Path,
                ymax: float | None = None, vms_n_runs: int | None = None,
                vms_bands: list[tuple[float, float, float]] | None = None) -> None:
    """vms_value is the MEDIAN CAROB over all VMS rows (vms_n_runs of them).
    vms_bands: [(coverage %, lo, hi), ...] nested central percentile bands of the
    VMS run values, drawn in shades of orange behind the median line."""
    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = matplotlib.colormaps["viridis"]
    colors = [cmap(p) for p in np.linspace(0.15, 0.85, len(groups))]

    parts = ax.violinplot([text_data[g] for g in groups],
                          positions=range(1, len(groups) + 1),
                          showmedians=True, showextrema=True, widths=0.6)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor("white")
        pc.set_alpha(0.75)

    rng = np.random.default_rng(0)
    for i, g in enumerate(groups):
        vals = text_data[g]
        ax.scatter(i + 1 + rng.uniform(-0.06, 0.06, size=len(vals)), vals,
                   color=colors[i], alpha=0.8, s=30, zorder=3,
                   edgecolors="white", linewidths=0.5)

    # Explicit y-limits (not left to autoscale): unless --ymax is given, the axis
    # always extends far enough to include the VMS value.
    vms_color = "#e05c2a"
    data_top = max(float(np.max(text_data[g])) for g in groups)
    if vms_value is not None:
        data_top = max(data_top, vms_value)
        for _c, _lo, hi in (vms_bands or []):
            data_top = max(data_top, hi)
    top = ymax if ymax is not None else data_top * 1.06
    ax.set_ylim(bottom=0, top=top)
    off_scale = vms_value is not None and vms_value >= top

    random_line = ax.axhline(1.0, color="#888", linestyle=":", linewidth=1.2, zorder=1,
                             label="random ordering (CAROB = 1)")
    median_handle, band_handles = None, []
    if vms_value is not None and vms_bands:
        # Nested bands, widest (palest) first; narrower bands are drawn darker.
        shown = [(c, lo, hi) for c, lo, hi in vms_bands if lo < top]
        alphas = np.linspace(0.13, 0.30, len(shown)) if len(shown) > 1 else [0.22]
        for (c, lo, hi), alpha in zip(shown, alphas):
            h = ax.axhspan(lo, hi, facecolor=vms_color, alpha=float(alpha), linewidth=0,
                           zorder=0.6, label=f"VMS central {c:g}% of runs ({lo:.2f}\u2013{hi:.2f})")
            band_handles.append(h)
        band_handles.reverse()          # legend: narrowest band first
    if vms_value is not None:
        runs = f", {vms_n_runs} runs" if vms_n_runs else ""
        vms_label = f"VMS median CAROB ({vms_value:.2f}{runs})"
        if off_scale:
            vms_label += " - off scale"
        median_handle = ax.axhline(vms_value, color=vms_color, linestyle="--", linewidth=1.5,
                                   alpha=0.9, zorder=2, label=vms_label)
        if off_scale:
            # The line itself is outside the axes; mark it at the top edge instead
            # of letting it vanish silently.
            x_mid = (len(groups) + 1) / 2
            ax.plot([x_mid], [top], marker="^", color=vms_color, markersize=12,
                    clip_on=False, zorder=6, linestyle="none")
            ax.annotate(f"VMS median = {vms_value:.2f} (off scale)", xy=(x_mid, top),
                        xytext=(0, -12), textcoords="offset points", ha="center",
                        va="top", color=vms_color, fontsize=10, fontweight="bold")
    # Legend sits above the axes so it can never cover data or the VMS line.
    handles = ([median_handle] if median_handle is not None else []) + band_handles + [random_line]
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.01), ncol=2,
              fontsize=9, frameon=False)

    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels([f"{g}\n(N={len(text_data[g])} texts)" for g in groups], fontsize=10)
    ax.set_ylabel("CAROB: conjoint adjacency ratio over baseline\n(median per source text)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path,
                    help="sweep_v2.py's results_full_all.csv")
    ap.add_argument("--vms-input", type=Path, default=None,
                    help="optional separate CSV (same schema) containing the VMS runs")
    ap.add_argument("--vms-pattern", default=VMS_PATTERN_DEFAULT,
                    help="regex (case-insensitive) identifying VMS rows by source_text")
    ap.add_argument("--output-dir", type=Path, default=Path("violin_v2_out"))
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--n-boot", type=int, default=N_BOOT_DEFAULT)
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--wpf", type=int, nargs="+", default=None,
                    help="only use these words_per_folio values (VMS rows with no "
                         "words_per_folio value are exempt)")
    ap.add_argument("--metrics", nargs="+", default=None,
                    help="only use these distance metrics")
    ap.add_argument("--best-cost-only", action="store_true",
                    help="only use each cell's lowest-seriation-cost restart")
    ap.add_argument("--vms-bands", type=float, nargs="+", default=VMS_BANDS_DEFAULT,
                    metavar="PCT", help="central coverage (%%) of the nested VMS percentile bands "
                    "drawn around the median, e.g. 50 90 (= IQR and 5th-95th percentile); "
                    "default: 50 90")
    ap.add_argument("--no-vms-bands", action="store_true", help="draw only the VMS median")
    ap.add_argument("--ymax", type=float, default=None,
                    help="y-axis upper limit (default: automatic, always including VMS). If the "
                         "VMS value is above it, VMS is marked at the top edge as off scale")
    args = ap.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load, filter, split ------------------------------------------------
    raw = read_results(args.input)
    vms_extra = read_results(args.vms_input) if args.vms_input else None
    n_rows_raw = len(raw)

    # Split first, then filter each side: --wpf describes the synthetic foliations,
    # so VMS rows without a words_per_folio value are exempt from it (--metrics and
    # --best-cost-only apply to both).
    baseline, vms = split_baseline_vms(raw, args.vms_pattern, vms_extra)
    baseline = apply_filters(baseline, args.wpf, args.metrics, args.best_cost_only)
    if baseline.empty:
        raise SystemExit("No baseline rows left after filtering.")
    if vms is not None:
        n_vms_before = len(vms)
        vms = apply_filters(vms, args.wpf, args.metrics, args.best_cost_only,
                            exempt_missing_wpf=True)
        if vms.empty:
            print(f"  WARNING: all {n_vms_before} VMS row(s) were removed by the filters "
                  f"(--metrics/--best-cost-only/--wpf). Continuing WITHOUT VMS -- check that "
                  f"the VMS run used the metrics you selected.")
            vms = None
        elif len(vms) < n_vms_before:
            print(f"  NOTE: filters kept {len(vms)} of {n_vms_before} VMS rows.")
    check_duplicates(baseline, args.input.name)

    baseline["bpq_label"] = baseline["bifolios_per_quire"].map(bpq_label)
    bpq_values = sorted(baseline["bifolios_per_quire"].unique())
    groups = [bpq_label(n) for n in bpq_values]
    has_vms = vms is not None

    print("=" * 78)
    print("Text-level (one point per source text) analysis of CAROB")
    print("=" * 78)
    print(f"Input: {args.input}  ({n_rows_raw} usable rows)")
    filt = []
    if args.wpf:
        filt.append(f"wpf in {args.wpf}")
    if args.metrics:
        filt.append(f"metric in {args.metrics}")
    if args.best_cost_only:
        filt.append("best-cost restart only")
    print("Filters: " + (", ".join(filt) if filt else "none (all windows/wpf/metrics/restarts)"))
    n_texts = baseline["source_text"].nunique()
    print(f"Baseline: {n_texts} source texts, {len(groups)} quire-construction groups "
          f"({', '.join(groups)}), {len(baseline)} rows")
    if len(groups) < 2:
        raise SystemExit("Need at least 2 quire-construction groups to compare.")

    vms_value = None
    if has_vms:
        vnames = sorted(vms["source_text"].unique())
        print(f"VMS: {len(vnames)} source name(s) {vnames}, {len(vms)} rows "
              f"(collapses to N=1 independent unit)")
        if len(vnames) > 1:
            print("  WARNING: several VMS source names found; all rows are pooled into one value.")
        if vms["bifolios_per_quire"].nunique() > 1:
            print(f"  WARNING: VMS rows span {vms['bifolios_per_quire'].nunique()} "
                  f"bifolios_per_quire values and are pooled. That is only meaningful "
                  f"if VMS used its real foliation rather than synthetic quires.")
        vms_value = float(vms[VALUE_COL].median())
    else:
        print("No VMS data found -- baseline-only analysis.")
    print()

    # ---- Aggregate to text level -------------------------------------------
    baseline_text = text_level_aggregate(baseline)
    text_data = {g: baseline_text.loc[baseline_text["bpq_label"] == g,
                                      "text_level_median"].values for g in groups}
    out_values = baseline_text
    if has_vms:
        vms_row = pd.DataFrame([dict(
            bpq_label=VMS_LABEL, source_text="+".join(vnames),
            text_level_median=vms_value, n_rows=len(vms),
            q25=vms[VALUE_COL].quantile(0.25), q75=vms[VALUE_COL].quantile(0.75),
            **({"n_windows": vms["window_idx"].nunique()} if "window_idx" in vms else {}))])
        out_values = pd.concat([baseline_text, vms_row], ignore_index=True)
    out_values.to_csv(out_dir / "text_level_values.csv", index=False)

    print("Text-level CAROB per group:")
    for g in groups:
        v = text_data[g]
        print(f"  {g:12s} N={len(v):3d}  median={np.median(v):7.3f}  "
              f"range=[{np.min(v):.2f}, {np.max(v):.2f}]")
    if has_vms:
        print(f"  {VMS_LABEL:12s} N=1    value={vms_value:7.3f}  (median CAROB over {len(vms)} VMS runs)")
    print()

    # ---- Paired tests ---------------------------------------------------------
    pivot = build_paired_matrix(baseline_text, groups)

    print("Pairwise Wilcoxon signed-rank (text-level, paired, Holm-corrected):")
    print("  [Each text is seriated under every quire construction, so groups are")
    print("   matched samples, not independent -- paired tests are used throughout.]")
    wil = run_pairwise_wilcoxon(pivot, groups, alpha=args.alpha)
    print(wil[["group_a", "group_b", "n_pairs", "statistic", "p_raw", "p_holm",
               "effect_size_rank_biserial", "significant"]].to_string(index=False))
    wil.to_csv(out_dir / "pairwise_wilcoxon_text_level.csv", index=False)

    print("\nPairwise paired bootstrap (resample texts, median of paired diffs, Holm-corrected):")
    boot = bootstrap_pairwise_median_diff(pivot, groups, n_boot=args.n_boot,
                                          seed=args.seed, alpha=args.alpha)
    print(boot[["group_a", "group_b", "n_pairs", "observed_median_diff", "boot_ci_lo",
                "boot_ci_hi", "p_bootstrap", "p_bootstrap_holm",
                "significant"]].to_string(index=False))
    boot.to_csv(out_dir / "pairwise_bootstrap_text_level.csv", index=False)

    if len(groups) >= 3:
        fr_stat, fr_p = stats.friedmanchisquare(*[pivot[g].values for g in groups])
        fr_line = (f"Friedman omnibus across {len(groups)} groups (N={len(pivot)} paired "
                   f"texts): chi2={fr_stat:.3f}, df={len(groups) - 1}, p={fr_p:.6g}")
    else:
        fr_line = "Friedman omnibus skipped: needs at least 3 groups."
    print(f"\n{fr_line}")
    (out_dir / "friedman.txt").write_text(fr_line + "\n")

    # ---- VMS membership -------------------------------------------------------
    if has_vms:
        print("\n" + "=" * 78)
        print("VMS group-membership report")
        print("=" * 78)
        print("  (a) exact rank test         -- nonparametric, floored at 2/(n+1)")
        print("  (b) empirical tail fraction -- simple, floors at 0/n")
        print("  (c) AIC-selected parametric fit + bootstrap CI of P(draw >= VMS) -- can go")
        print("      below 1/(n+1) but assumes a parametric shape; see")
        print("      distribution_fit_comparison.csv for every family's AIC\n")
        report, aic = vms_membership_report(vms_value, baseline_text, groups,
                                            n_boot=args.n_boot, seed=args.seed)
        with pd.option_context("display.float_format", lambda x: f"{x:.4g}"):
            print(report.to_string(index=False))
        report.to_csv(out_dir / "vms_membership_report.csv", index=False)
        if not aic.empty:
            print("\nPer-group distribution fit comparison (AIC, lower is better; "
                  "KS p-values optimistic):")
            with pd.option_context("display.float_format", lambda x: f"{x:.4g}"):
                print(aic.to_string(index=False))
            aic.to_csv(out_dir / "distribution_fit_comparison.csv", index=False)

    # ---- Plot + settings --------------------------------------------------------
    if vms_value is not None and args.ymax is not None and vms_value >= args.ymax:
        print(f"  WARNING: VMS CAROB ({vms_value:.2f}) is at/above --ymax ({args.ymax:g}); "
              f"it is marked at the top edge of the plot as off scale rather than drawn as a line.")
    vms_bands = None
    if has_vms:
        bad = [c for c in args.vms_bands if not 0 < c < 100]
        if bad:
            raise SystemExit(f"--vms-bands values must be between 0 and 100 (got {bad}).")
        vals = vms[VALUE_COL].to_numpy(dtype=float)
        stat_rows = [("n_runs", float(len(vals))), ("min", float(vals.min())),
                     ("median", float(np.median(vals))), ("max", float(vals.max()))]
        if "metric" in vms.columns:
            stat_rows += [(f"median[metric={m}]", float(g[VALUE_COL].median()))
                          for m, g in vms.groupby("metric")]
        if not args.no_vms_bands:
            if len(vals) >= MIN_RUNS_FOR_BANDS:
                vms_bands = vms_band_table(vals, args.vms_bands)
                for c, lo, hi in vms_bands:
                    stat_rows += [(f"central_{c:g}pct_lo", lo), (f"central_{c:g}pct_hi", hi)]
                print("\nVMS run-to-run spread (percentile bands, median "
                      f"{vms_value:.2f}):  " + "   ".join(
                          f"central {c:g}%: {lo:.2f}-{hi:.2f}" for c, lo, hi in vms_bands))
            else:
                print(f"\n  NOTE: only {len(vals)} VMS run(s) (< {MIN_RUNS_FOR_BANDS}); "
                      f"percentile bands not drawn.")
        pd.DataFrame(stat_rows, columns=["statistic", "value"]).to_csv(
            out_dir / "vms_percentiles.csv", index=False)

    svg = out_dir / "violin_text_level.svg"
    make_violin(text_data, groups, vms_value, svg, ymax=args.ymax,
                vms_n_runs=len(vms) if has_vms else None, vms_bands=vms_bands)
    print(f"\nPlot -> {svg}")

    (out_dir / "run_settings.json").write_text(json.dumps(dict(
        input=str(args.input), vms_input=str(args.vms_input) if args.vms_input else None,
        vms_pattern=args.vms_pattern, value_column=VALUE_COL,
        filters=dict(wpf=args.wpf, metrics=args.metrics, best_cost_only=args.best_cost_only),
        rows_input=n_rows_raw, rows_baseline=len(baseline),
        rows_vms=len(vms) if has_vms else 0,
        n_texts=n_texts, groups=groups, alpha=args.alpha,
        n_boot=args.n_boot, seed=args.seed,
        vms_bands=None if args.no_vms_bands else args.vms_bands,
        versions=dict(pandas=pd.__version__, numpy=np.__version__, scipy=scipy.__version__,
                      matplotlib=matplotlib.__version__),
    ), indent=2))
    print("Done.")


if __name__ == "__main__":
    main()
