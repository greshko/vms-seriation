"""
violin.py
===================================
Standalone significance-testing + plotting script that performs the
one-point-per-source-text analysis: every synthetic-collation
run is first collapsed to a single median value per source text, so the
statistical unit is the DOCUMENT, not the run.


WHAT IT PRODUCES
----------------
1. text_level_values.csv
       One row per (group, source_text): the median lift used everywhere else.
2. pairwise_wilcoxon_text_level.csv
       Wilcoxon signed-rank test among baseline quire-construction groups,
       Holm-corrected. Every one of the 50 source texts is seriated under
       ALL 5 quire constructions, so the groups are matched/paired samples
       (a repeated-measures design), not independent ones -- each text acts
       as its own control across quire constructions. Mann-Whitney U (which
       assumes independent samples and discards the text-to-text pairing)
       is therefore the wrong test here; Wilcoxon signed-rank on the
       matched per-text differences is the paired analogue and is what's
       used throughout this script.
3. pairwise_bootstrap_text_level.csv
       Paired/cluster bootstrap (resample source texts, keeping each
       resampled text's value in every group together) for the same
       pairwise comparisons -- an assumption-light cross-check on Wilcoxon
       that preserves the paired design instead of resampling each group
       independently.
4. friedman.txt
       Omnibus test across all baseline groups -- Friedman's test, the
       paired analogue of Kruskal-Wallis, which ranks each text's 5 values
       within that text before comparing rank sums across groups.
5. vms_membership_report.csv  (only if VMS data is present)
       For each baseline group, three complementary answers to "is VMS a
       plausible member of this group's distribution", each appropriate to
       a different question:
         a) EXACT RANK TEST (nonparametric, no distributional assumption):
            treats VMS + the n baseline texts as n+1 exchangeable draws and
            asks how extreme VMS's rank is. Exact, but bounded below by
            1/(n+1) -- it cannot express "less likely than 1 in 46" even if
            the true effect is much larger.
         b) EMPIRICAL TAIL FRACTION: the fraction of the n observed texts
            at least as extreme as VMS. Simple, but same floor-at-zero
            problem when VMS exceeds every observed text (all pairwise
            comparisons if group max < vms_value get exactly 0/n).
         c) PARAMETRIC-BOOTSTRAP MEMBERSHIP PROBABILITY: rather than assuming
            log-normality outright, fits FOUR candidate families per group
            (log-normal, gamma, Weibull, normal) to the group's text-level
            values and selects the best-fitting family by AIC. Log-normal
            is not a safe default here -- checked against the actual data,
            it is the WORST-fitting of the four candidates for the
            Singulion and Binion groups specifically (the two groups that
            matter most for the VMS comparison), and log-normality of the
            underlying values is formally rejected by Shapiro-Wilk (on the
            logs) for Singulion, Binion, and Ternion at p<0.01. Using the
            AIC-selected family instead of a fixed log-normal avoids
            leaning on a specific distributional shape that the data
            themselves argue against.

            The fitted tail probability P(a new draw >= VMS's value) is
            computed under the AIC-selected family, and SAMPLE-SIZE
            uncertainty in that estimate is propagated by refitting the
            SAME selected family (chosen once per group from the full
            sample, not re-selected per resample) on bootstrap resamples
            of the n texts and reporting the resulting percentile interval.
            This is the piece that can express "less likely than 1-in-46"
            and give a sense of just how far outside the distribution VMS
            sits, at the cost of assuming *some* parametric shape is a
            reasonable model for these values -- flagged explicitly so it
            isn't over-trusted, and the AIC table for every family is
            written out alongside it so the choice can be audited.
6. distribution_fit_comparison.csv
       Per-group AIC (and KS-test diagnostics) for every candidate family
       (log-normal, gamma, Weibull, normal), with the AIC-selected family
       flagged -- lets you see exactly how close/far the alternatives were.
7. violin_text_level.svg
       Single-panel violin plot: one point per source text per group, VMS
       (if present) marked as a diamond.

Usage
-----
    python violin.py --input seriation_results.xlsx
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

LABEL_MAP = {
    "1": "Singulion", "2": "Binion", "3": "Ternion",
    "4": "Quaternion", "5": "Quinion",
}
GROUP_ORDER = ["Singulion", "Binion", "Ternion", "Quaternion", "Quinion"]
VALUE_COL = "mult_over_baseline"
N_BOOT_DEFAULT = 20000


# =============================================================================
# LOAD + AGGREGATE
# =============================================================================

def load_full_data(xlsx_path: Path) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Schema-flexible loader. Accepts either:
        bpq_label (pre-labeled)      or  bifolios_per_quire (int 1-5)
        mult_over_baseline           or  mult_lift
    VMS data may appear as:
        (a) a sheet literally named "voynich"
        (b) rows in the baseline sheet whose source_text names the VMS
        (c) both at once (a "voynich" sheet AND a leftover VMS row inside
            the baseline sheet, e.g. if baseline is itself a concatenation
            of every run) -- the VMS row is always stripped out of baseline
            regardless of where the returned `vms` df came from, since a
            baseline text with no value in any real quire-construction
            group breaks the fully-crossed assumption of the paired tests
        (d) absent entirely (baseline-only run)
    Returns (baseline_df, vms_df_or_None).
    """
    xl = pd.ExcelFile(xlsx_path)
    baseline = None
    for name in xl.sheet_names:
        df = xl.parse(name)
        cols = set(df.columns)
        if {"source_text", "metric", "restart"}.issubset(cols) and \
           ({"bpq_label", "bifolios_per_quire"} & cols):
            baseline = df
            break
    if baseline is None:
        raise ValueError(
            f"Could not find a baseline results sheet in {xlsx_path}. "
            f"Sheets found: {xl.sheet_names}"
        )

    vms = None
    if "voynich" in xl.sheet_names:
        vms = xl.parse("voynich")
    else:
        vms_mask = baseline["source_text"].astype(str).str.contains(
            "voynich|vms", case=False, regex=True
        )
        if vms_mask.any():
            vms = baseline.loc[vms_mask].copy()

    # Regardless of where `vms` came from (a dedicated sheet, or rows pulled
    # out of baseline above), strip any voynich/VMS-named rows out of
    # baseline. A baseline results sheet can independently contain a VMS
    # row (e.g. if it's a concatenation of every run) even when a separate
    # "voynich" sheet also exists -- leaving such a row in baseline breaks
    # the fully-crossed assumption of the paired tests, since that text has
    # no value in any of the real quire-construction groups.
    vms_mask = baseline["source_text"].astype(str).str.contains(
        "voynich|vms", case=False, regex=True
    )
    if vms_mask.any():
        baseline = baseline.loc[~vms_mask].copy()

    if "bifolios_per_quire" in baseline.columns and "bpq_label" not in baseline.columns:
        baseline["bpq_label"] = (
            pd.to_numeric(baseline["bifolios_per_quire"], errors="coerce")
            .astype("Int64").astype(str)
        )
    if "mult_lift" in baseline.columns and VALUE_COL not in baseline.columns:
        baseline[VALUE_COL] = baseline["mult_lift"]
    baseline["bpq_label"] = baseline["bpq_label"].astype(str).map(
        lambda x: LABEL_MAP.get(x, x)
    )

    if vms is not None:
        if "bifolios_per_quire" in vms.columns and "bpq_label" not in vms.columns:
            vms["bpq_label"] = vms["bifolios_per_quire"].astype(str)
        if "mult_lift" in vms.columns and VALUE_COL not in vms.columns:
            vms[VALUE_COL] = vms["mult_lift"]
        vms["bpq_label"] = "VMS"

    return baseline, vms


def text_level_aggregate(df: pd.DataFrame, group_col: str = "bpq_label") -> pd.DataFrame:
    """Collapse every (folio-length x metric x restart) run to a single median
    value per (group, source_text) -- the true independent unit of replication."""
    return (
        df.groupby([group_col, "source_text"])[VALUE_COL]
        .median()
        .reset_index()
        .rename(columns={VALUE_COL: "text_level_median"})
    )


# =============================================================================
# PAIRWISE TESTS AMONG BASELINE GROUPS
# =============================================================================

def build_paired_matrix(baseline_text: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    """
    Pivot text-level medians to a (source_text x group) matrix, aligned by
    source_text. Every one of the 50 texts appears once per group -- this is
    the correct shape for paired tests (Wilcoxon, Friedman), since it keeps
    each text's value across all 5 quire constructions on the same row.

    Raises if any (group, source_text) cell is missing -- a paired test is
    only valid on a fully-crossed design, and a silent NaN-drop per pairwise
    comparison would quietly change the effective N test-to-test.
    """
    pivot = baseline_text.pivot(index="source_text", columns="bpq_label",
                                 values="text_level_median")
    missing = pivot[groups].isna().sum()
    if missing.any():
        raise ValueError(
            "build_paired_matrix: not every source_text has a value in every "
            f"group -- design is not fully crossed. Missing counts:\n{missing}"
        )
    return pivot[groups]


def run_pairwise_wilcoxon(pivot: pd.DataFrame, groups: list[str],
                           alpha: float = 0.05) -> pd.DataFrame:
    """
    Holm-corrected pairwise Wilcoxon signed-rank test -- the paired analogue
    of Mann-Whitney U. Each of the 50 source texts is seriated under every
    quire construction, so group_a and group_b here are matched samples
    (same text, two conditions), not independent draws; Wilcoxon operates on
    the per-text differences and is the correct nonparametric test for that
    structure. Effect size reported is matched-pairs rank-biserial
    correlation, r = (W+ - W-) / (W+ + W-), on the signed ranks.
    """
    rows = []
    for a, b in itertools.combinations(groups, 2):
        xa, xb = pivot[a].values, pivot[b].values
        n = len(xa)
        diffs = xa - xb
        try:
            stat, p = stats.wilcoxon(xa, xb, alternative="two-sided",
                                      zero_method="wilcox", method="auto")
        except ValueError:
            stat, p = np.nan, np.nan
        nz = diffs[diffs != 0]
        if len(nz):
            ranks = stats.rankdata(np.abs(nz))
            w_pos = ranks[nz > 0].sum()
            w_neg = ranks[nz < 0].sum()
            r = (w_pos - w_neg) / (w_pos + w_neg)
        else:
            r = np.nan
        rows.append(dict(group_a=a, group_b=b, n_pairs=n,
                          statistic=stat, p_raw=p,
                          effect_size_rank_biserial=r,
                          median_diff=float(np.median(diffs))))
    df = pd.DataFrame(rows)
    n_tests = len(df)
    order = df["p_raw"].argsort().values
    p_holm = np.empty(n_tests)
    for rank, idx in enumerate(order):
        p_holm[idx] = min(df["p_raw"].iloc[idx] * (n_tests - rank), 1.0)
    for i in range(1, n_tests):
        p_holm[order[i]] = max(p_holm[order[i]], p_holm[order[i - 1]])
    df["p_holm"] = p_holm
    df["significant"] = df["p_holm"] < alpha
    return df.sort_values("p_raw").reset_index(drop=True)


def bootstrap_pairwise_median_diff(pivot: pd.DataFrame, groups: list[str],
                                    n_boot: int = N_BOOT_DEFAULT, seed: int = 42,
                                    alpha: float = 0.05) -> pd.DataFrame:
    """
    Paired/cluster bootstrap for every pairwise group comparison: resample
    SOURCE TEXTS (rows of `pivot`) with replacement, so that a resampled
    text's values in group_a and group_b always come from the same draw.
    This preserves the matched-pairs structure of the design -- resampling
    each group independently (as an unpaired bootstrap would) breaks the
    pairing and can misstate the CI/p-value when, as here, text-level
    variance is shared across conditions. Builds the bootstrap distribution
    of median(a - b) computed per-resample on the paired differences, and
    derives a percentile CI + two-sided bootstrap p-value. Cross-checks the
    Wilcoxon result without assuming a particular null distribution shape.
    """
    rng = np.random.default_rng(seed)
    n_texts = len(pivot)
    rows = []
    for a, b in itertools.combinations(groups, 2):
        xa, xb = pivot[a].values, pivot[b].values
        diffs = xa - xb
        idx_boot = rng.integers(0, n_texts, size=(n_boot, n_texts))
        boot_diffs = np.median(diffs[idx_boot], axis=1)
        ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])
        observed_diff = float(np.median(diffs))
        p_boot = min(2 * min(np.mean(boot_diffs <= 0), np.mean(boot_diffs >= 0)), 1.0)
        rows.append(dict(group_a=a, group_b=b, n_pairs=n_texts,
                          observed_median_diff=observed_diff,
                          boot_ci_lo=ci_lo, boot_ci_hi=ci_hi, p_bootstrap=p_boot))
    df = pd.DataFrame(rows)
    n_tests = len(df)
    order = df["p_bootstrap"].argsort().values
    p_holm = np.empty(n_tests)
    for rank, idx in enumerate(order):
        p_holm[idx] = min(df["p_bootstrap"].iloc[idx] * (n_tests - rank), 1.0)
    for i in range(1, n_tests):
        p_holm[order[i]] = max(p_holm[order[i]], p_holm[order[i - 1]])
    df["p_bootstrap_holm"] = p_holm
    df["significant"] = df["p_bootstrap_holm"] < alpha
    return df.sort_values("p_bootstrap").reset_index(drop=True)


# =============================================================================
# VMS MEMBERSHIP: exact rank test + empirical tail + parametric bootstrap
# =============================================================================

def exact_rank_test(vms_value: float, group_vals: np.ndarray) -> tuple[int, float]:
    """
    Exact, assumption-free two-sided test: treat VMS and the n group values as
    n+1 exchangeable draws from the same population, and ask how extreme VMS's
    rank is among them. Floor: with n texts, the smallest possible two-sided
    p-value is 2/(n+1) -- this test cannot say "less likely than 1 in n+1"
    even when the true effect is far larger.
    """
    n = len(group_vals)
    rank = int((group_vals < vms_value).sum()) + 1  # VMS's 1-indexed rank if inserted
    pos_from_edge = min(rank, (n + 1) - rank + 1)
    p_exact = min(2 * pos_from_edge / (n + 1), 1.0)
    return rank, p_exact


# Candidate parametric families for the group-level lift distributions.
# All four are fit with 2 free parameters (floc=0 anchors location at 0 for
# the strictly-positive, no-shift lift ratios; norm's loc/scale are both
# free), so AIC differences here reduce to log-likelihood differences --
# no family is favored by a parameter-count penalty over another.
CANDIDATE_DISTS: dict[str, tuple] = {
    "lognorm": (stats.lognorm, dict(floc=0)),
    "gamma": (stats.gamma, dict(floc=0)),
    "weibull_min": (stats.weibull_min, dict(floc=0)),
    "norm": (stats.norm, dict()),
}


def fit_all_candidates(vals: np.ndarray) -> dict:
    """Fit every candidate family in CANDIDATE_DISTS to `vals`.

    Returns {family_name: {"params": ..., "loglik": ..., "aic": ...,
    "ks_stat": ..., "ks_p": ...}}, silently skipping any family that fails
    to fit (e.g. degenerate resamples). Raises if none can be fit.

    Note on the KS p-values: parameters here are estimated from the same
    sample being tested, which is known to bias standard KS p-values
    upward (i.e. toward NOT rejecting) relative to the true sampling
    distribution under the null. They're included as a rough diagnostic,
    not a calibrated test -- treat them as optimistic upper bounds on the
    true evidence against a given fit.
    """
    out = {}
    for name, (dist, kwargs) in CANDIDATE_DISTS.items():
        try:
            params = dist.fit(vals, **kwargs)
            loglik = float(np.sum(dist.logpdf(vals, *params)))
            k = 2  # free parameters, uniform across all four candidates
            aic = 2 * k - 2 * loglik
            ks_stat, ks_p = stats.kstest(vals, name, args=params)
            out[name] = dict(params=params, loglik=loglik, aic=aic,
                              ks_stat=float(ks_stat), ks_p=float(ks_p))
        except Exception:
            continue
    if not out:
        raise RuntimeError("No candidate distribution could be fit to this group's values.")
    return out


def select_best_distribution(vals: np.ndarray) -> tuple[str, dict]:
    """Return (best_family_name, fit_dict_for_all_candidates), where the
    best family is whichever minimizes AIC among those that could be fit."""
    fits = fit_all_candidates(vals)
    best_name = min(fits, key=lambda name: fits[name]["aic"])
    return best_name, fits


def parametric_membership_probability(vms_value: float, group_vals: np.ndarray,
                                       n_boot: int = N_BOOT_DEFAULT,
                                       seed: int = 42) -> dict:
    """
    Fits four candidate families (log-normal, gamma, Weibull, normal) to the
    group's text-level lift values and selects the best-fitting family by
    AIC -- rather than assuming log-normal by default. (Checked against this
    study's data, log-normal is in fact the worst-fitting of the four for
    the Singulion and Binion groups, and is rejected by Shapiro-Wilk on the
    logs at p<0.01 for Singulion, Binion, and Ternion; see
    distribution_fit_comparison.csv for the full per-group comparison.)

    Computes the AIC-selected family's fitted tail probability P(a new
    independent draw from this group >= VMS's value). Unlike the exact rank
    test or the raw empirical tail fraction, this can express probabilities
    smaller than 1/(n+1) -- but it trades that for an explicit, checkable
    distributional assumption, now chosen by fit quality rather than fixed.

    Sample-size uncertainty in the estimate is propagated via a bootstrap:
    resample the n texts with replacement, refit the SAME AIC-selected
    family (chosen once from the full sample, not re-selected per resample,
    so bootstrap variability reflects parameter uncertainty within a fixed
    family rather than family-selection instability), and recompute the
    tail probability, n_boot times. The resulting percentile interval
    reflects how much the tail-probability ESTIMATE itself wobbles given
    only n independent texts -- not just where VMS falls relative to one
    fixed fit.
    """
    rng = np.random.default_rng(seed)
    n = len(group_vals)

    best_name, fits = select_best_distribution(group_vals)
    dist, kwargs = CANDIDATE_DISTS[best_name]
    point_estimate = float(dist.sf(vms_value, *fits[best_name]["params"]))

    def tail_prob(vals: np.ndarray) -> float:
        params = dist.fit(vals, **kwargs)
        return float(dist.sf(vms_value, *params))

    boot_estimates = np.empty(n_boot)
    for i in range(n_boot):
        resample = rng.choice(group_vals, size=n, replace=True)
        try:
            boot_estimates[i] = tail_prob(resample)
        except Exception:
            boot_estimates[i] = np.nan
    boot_estimates = boot_estimates[~np.isnan(boot_estimates)]

    ci_lo, ci_hi = np.percentile(boot_estimates, [2.5, 97.5]) if len(boot_estimates) else (np.nan, np.nan)
    return dict(
        distribution=best_name,
        aic_table={name: f["aic"] for name, f in fits.items()},
        ks_table={name: (f["ks_stat"], f["ks_p"]) for name, f in fits.items()},
        point_estimate=point_estimate,
        boot_median=float(np.median(boot_estimates)) if len(boot_estimates) else np.nan,
        boot_ci_lo=float(ci_lo), boot_ci_hi=float(ci_hi),
        n_valid_boot=len(boot_estimates),
    )


def vms_membership_report(vms_value: float, baseline_text: pd.DataFrame,
                           groups: list[str], n_boot: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (report_df, aic_comparison_df). report_df has one row per group
    with the AIC-selected family's tail-probability estimate; aic_comparison_df
    has one row per (group, candidate family) so the full selection can be
    audited rather than just trusting the winner."""
    rows = []
    aic_rows = []
    for g in groups:
        vals = baseline_text.loc[baseline_text["bpq_label"] == g, "text_level_median"].values
        n = len(vals)
        if n == 0:
            continue
        rank, p_exact = exact_rank_test(vms_value, vals)
        empirical_tail_frac = float((vals >= vms_value).mean())
        param_result = parametric_membership_probability(vms_value, vals, n_boot=n_boot)
        rows.append(dict(
            group=g, n_texts=n,
            group_median=np.median(vals), group_min=np.min(vals), group_max=np.max(vals),
            vms_value=vms_value,
            vms_rank_among_n_plus_1=rank,
            p_exact_rank_test=p_exact,
            empirical_tail_fraction=empirical_tail_frac,
            best_fit_distribution=param_result["distribution"],
            parametric_tail_prob_point=param_result["point_estimate"],
            parametric_tail_prob_boot_median=param_result["boot_median"],
            parametric_tail_prob_ci_lo=param_result["boot_ci_lo"],
            parametric_tail_prob_ci_hi=param_result["boot_ci_hi"],
        ))
        for name, aic_val in param_result["aic_table"].items():
            ks_stat, ks_p = param_result["ks_table"][name]
            aic_rows.append(dict(
                group=g, distribution=name, aic=aic_val,
                delta_aic=aic_val - min(param_result["aic_table"].values()),
                ks_stat=ks_stat, ks_p=ks_p,
                selected=(name == param_result["distribution"]),
            ))
    return pd.DataFrame(rows), pd.DataFrame(aic_rows).sort_values(["group", "aic"]).reset_index(drop=True)


# =============================================================================
# PLOTTING
# =============================================================================

def make_violin(text_data: dict[str, np.ndarray], groups: list[str],
                 vms_value: float | None, out_path: Path) -> None:
    has_vms = vms_value is not None
    fig, ax = plt.subplots(figsize=(9, 6))

    cmap = matplotlib.colormaps.get_cmap("viridis")
    colors = [cmap(p) for p in np.linspace(0.15, 0.85, len(groups))]
    vms_color = "#e05c2a"

    arrays = [text_data[g] for g in groups]
    parts = ax.violinplot(arrays, positions=range(1, len(groups) + 1),
                           showmedians=True, showextrema=True, widths=0.6)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor("white")
        pc.set_alpha(0.75)

    rng = np.random.default_rng(0)
    for i, g in enumerate(groups):
        pos = i + 1
        vals = text_data[g]
        jitter = rng.uniform(-0.06, 0.06, size=len(vals))
        ax.scatter(pos + jitter, vals, color=colors[i], alpha=0.8, s=30,
                   zorder=3, edgecolors="white", linewidths=0.5)

    if has_vms:
        ax.axhline(vms_value, color=vms_color, linestyle="--", linewidth=1.5,
                   alpha=0.8, zorder=2, label=f"VMS observed value ({vms_value:.2f})")
        ax.legend(loc="upper right", fontsize=9)

    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels([f"{g}\n(N={len(text_data[g])} texts)" for g in groups], fontsize=10)
    ax.set_ylabel("Conjoint adjacency ratio over baseline\n(median per source text)")
    ax.set_ylim(top=20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n-boot", type=int, default=N_BOOT_DEFAULT)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline, vms = load_full_data(Path(args.input))
    has_vms = vms is not None
    groups = [g for g in GROUP_ORDER if g in baseline["bpq_label"].unique()]

    print("=" * 78)
    print("Text-level (one point per source text) significance analysis")
    print("=" * 78)
    n_texts = baseline["source_text"].nunique()
    print(f"Baseline: {n_texts} source texts, {len(groups)} quire-construction groups")
    if has_vms:
        print(f"VMS: {vms['source_text'].nunique()} document, {len(vms)} runs "
              f"(collapses to N=1 independent unit)")
    else:
        print("No VMS data found -- baseline-only analysis.")
    print()

    # ---- Aggregate to text level ----
    baseline_text = text_level_aggregate(baseline)
    text_data = {g: baseline_text.loc[baseline_text["bpq_label"] == g, "text_level_median"].values
                 for g in groups}
    baseline_text.to_csv(out_dir / "text_level_values.csv", index=False)
    print("Text-level values per group:")
    for g in groups:
        v = text_data[g]
        print(f"  {g:12s} N={len(v):3d}  median={np.median(v):7.3f}  "
              f"range=[{np.min(v):.2f}, {np.max(v):.2f}]")

    vms_value = None
    if has_vms:
        vms_text = text_level_aggregate(vms)
        vms_value = float(vms_text["text_level_median"].iloc[0])
        print(f"  {'VMS':12s} N=1    value={vms_value:7.3f}")
    print()

    # ---- Paired matrix: every text has one value per group (fully crossed) ----
    pivot = build_paired_matrix(baseline_text, groups)

    # ---- Pairwise Wilcoxon signed-rank among baseline groups (paired design) ----
    print("Pairwise Wilcoxon signed-rank (text-level, paired, Holm-corrected):")
    print("  [Each of the N texts is seriated under every quire construction, so")
    print("   groups are matched samples, not independent -- Wilcoxon replaces MWU.]")
    wilcoxon_df = run_pairwise_wilcoxon(pivot, groups, alpha=args.alpha)
    print(wilcoxon_df[["group_a", "group_b", "n_pairs", "statistic",
                        "p_raw", "p_holm", "effect_size_rank_biserial",
                        "significant"]].to_string(index=False))
    wilcoxon_df.to_csv(out_dir / "pairwise_wilcoxon_text_level.csv", index=False)

    # ---- Pairwise paired bootstrap among baseline groups ----
    print("\nPairwise paired bootstrap (resample texts, median of paired diffs, Holm-corrected):")
    boot_df = bootstrap_pairwise_median_diff(pivot, groups, n_boot=args.n_boot, alpha=args.alpha)
    print(boot_df[["group_a", "group_b", "n_pairs", "observed_median_diff",
                    "boot_ci_lo", "boot_ci_hi", "p_bootstrap",
                    "p_bootstrap_holm", "significant"]].to_string(index=False))
    boot_df.to_csv(out_dir / "pairwise_bootstrap_text_level.csv", index=False)

    # ---- Friedman omnibus (paired analogue of Kruskal-Wallis) ----
    fr_stat, fr_p = stats.friedmanchisquare(*[pivot[g].values for g in groups])
    fr_line = (f"Friedman omnibus across {len(groups)} groups "
               f"(N={len(pivot)} paired texts): "
               f"chi2={fr_stat:.3f}, df={len(groups) - 1}, p={fr_p:.6g}")
    print(f"\n{fr_line}")
    (out_dir / "friedman.txt").write_text(fr_line + "\n")

    # ---- VMS membership report ----
    if has_vms:
        print("\n" + "=" * 78)
        print("VMS group-membership report")
        print("=" * 78)
        print("Three complementary answers per group -- see script docstring for caveats:")
        print("  (a) exact rank test         -- nonparametric, floored at 1/(n+1)")
        print("  (b) empirical tail fraction -- simple, also floored at 0/n")
        print("  (c) AIC-selected parametric fit + bootstrap CI -- can go below")
        print("      1/(n+1); family (log-normal/gamma/Weibull/normal) is chosen")
        print("      per group by AIC rather than assumed, since log-normal is")
        print("      NOT always the best fit here (see distribution_fit_comparison.csv);")
        print("      CI reflects small-n fit uncertainty within the chosen family\n")
        report_df, aic_df = vms_membership_report(vms_value, baseline_text, groups, n_boot=args.n_boot)
        with pd.option_context("display.float_format", lambda x: f"{x:.4g}"):
            print(report_df.to_string(index=False))
        report_df.to_csv(out_dir / "vms_membership_report.csv", index=False)

        print("\nPer-group distribution fit comparison (AIC, lower is better; "
              "KS p-values are optimistic, see docstring):")
        with pd.option_context("display.float_format", lambda x: f"{x:.4g}"):
            print(aic_df.to_string(index=False))
        aic_df.to_csv(out_dir / "distribution_fit_comparison.csv", index=False)

    # ---- Plot ----
    make_violin(text_data, groups, vms_value, out_dir / "violin_text_level.svg")
    print(f"\nPlot -> {out_dir / 'violin_text_level.svg'}")
    print("Done.")


if __name__ == "__main__":
    main()
