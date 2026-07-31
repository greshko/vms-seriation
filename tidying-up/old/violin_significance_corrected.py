"""
violin_significance_corrected.py
=================================
Corrected violin plots + significance tests for the bifolium adjacency
multiplicative-lift data, fixing the pseudoreplication problem in the
original analysis.

THE PROBLEM WITH THE ORIGINAL ANALYSIS
---------------------------------------
The original script (violin_significance.py) ran Mann-Whitney U tests
treating each of the 576 "baseline" rows per quire-construction group as an
independent observation. But those 576 rows are:

    8 source texts  x  3 folio lengths  x  3 distance metrics  x  8 SA restarts

i.e. 576 *correlated* re-analyses of only 8 underlying documents. Likewise
the 24 "VMS" rows are 3 metrics x 8 restarts applied to a *single* manuscript
-- not 24 independent manuscripts. Feeding these directly into Mann-Whitney U
massively inflates the effective sample size, which is why the original
analysis produced p-values as small as 1e-105.

THE FIX
-------
Aggregate every run down to one value per truly-independent unit -- the
source text -- *before* running any significance test:

    * Baseline groups: 8 source texts -> 8 independent values per quire
      construction (median mult_over_baseline across all folio-lengths x
      metrics x restarts for that text).
    * VMS: only ONE source text exists, so it collapses to N=1. There is no
      valid two-sample frequentist test against a group with N=1 -- this
      script reports that plainly rather than manufacturing a p-value, and
      instead reports VMS's percentile rank against each group's 8
      text-level values plus a bootstrap-based comparison.

This script performs, and reports side-by-side:
    1. The ORIGINAL (pseudoreplicated) analysis, for comparison.
    2. The CORRECTED analysis at the text level (n=8 per baseline group).
    3. A cluster-aware treatment of VMS-vs-baseline comparisons.

Usage
-----
    python violin_significance_corrected.py --input all_seriation_results.xlsx
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
    "1": "Singulion", "2": "Binion", "3": "Trinion",
    "4": "Quaternion", "5": "Quinion",
}
GROUP_ORDER = ["VMS", "Singulion", "Binion", "Trinion", "Quaternion", "Quinion"]
VALUE_COL = "mult_over_baseline"


# =============================================================================
# LOAD + RESHAPE
# =============================================================================

def load_full_data(xlsx_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the baseline ('all') and VMS ('voynich') sheets with full metadata."""
    baseline = pd.read_excel(xlsx_path, sheet_name="all")
    vms = pd.read_excel(xlsx_path, sheet_name="voynich")

    baseline["bpq_label"] = baseline["bpq_label"].astype(str).map(
        lambda x: LABEL_MAP.get(x, x)
    )
    vms["bpq_label"] = "VMS"

    return baseline, vms


def text_level_aggregate(df: pd.DataFrame, group_col: str = "bpq_label") -> pd.DataFrame:
    """
    Collapse every (folio-length x metric x restart) run down to a single
    median value per (group, source_text). This is the true independent
    unit of replication: one number per document.
    """
    agg = (
        df.groupby([group_col, "source_text"])[VALUE_COL]
        .median()
        .reset_index()
        .rename(columns={VALUE_COL: "text_level_median"})
    )
    return agg


# =============================================================================
# ORIGINAL (PSEUDOREPLICATED) ANALYSIS -- for side-by-side comparison only
# =============================================================================

def run_pairwise_mwu(data: dict[str, np.ndarray], groups: list[str],
                      correction: str = "bonferroni", alpha: float = 0.05,
                      label: str = "", method: str = "auto") -> pd.DataFrame:
    pairs = list(itertools.combinations(groups, 2))
    rows = []
    for a, b in pairs:
        xa, xb = data[a], data[b]
        if len(xa) < 1 or len(xb) < 1:
            continue
        try:
            stat, p = stats.mannwhitneyu(xa, xb, alternative="two-sided", method=method)
        except ValueError:
            stat, p = np.nan, np.nan
        n1, n2 = len(xa), len(xb)
        r = 1 - (2 * stat) / (n1 * n2) if n1 and n2 and not np.isnan(stat) else np.nan
        rows.append(dict(group_a=a, group_b=b, n_a=n1, n_b=n2,
                          statistic=stat, p_raw=p, effect_size_r=r))
    df = pd.DataFrame(rows)
    n_tests = len(df)
    if correction == "bonferroni" and n_tests:
        df["p_corrected"] = np.minimum(df["p_raw"] * n_tests, 1.0)
    elif correction == "holm" and n_tests:
        order = df["p_raw"].argsort().values
        p_holm = np.empty(n_tests)
        for rank, idx in enumerate(order):
            p_holm[idx] = min(df["p_raw"].iloc[idx] * (n_tests - rank), 1.0)
        for i in range(1, n_tests):
            p_holm[order[i]] = max(p_holm[order[i]], p_holm[order[i - 1]])
        df["p_corrected"] = p_holm
    else:
        df["p_corrected"] = df["p_raw"]
    df["significant"] = df["p_corrected"] < alpha
    df["analysis"] = label
    return df.sort_values("p_corrected").reset_index(drop=True)


# =============================================================================
# CORRECTED, CLUSTER-AWARE ANALYSIS
# =============================================================================

def minimum_achievable_p(n1: int, n2: int) -> float:
    """Smallest two-sided p-value Mann-Whitney U can ever produce for given n."""
    if n1 < 1 or n2 < 1:
        return np.nan
    xa = np.arange(n1)
    xb = np.arange(n1, n1 + n2) + 1000  # fully separated -> most extreme case
    _, p = stats.mannwhitneyu(xa, xb, alternative="two-sided")
    return p


def vms_percentile_report(vms_value: float, baseline_text_level: pd.DataFrame,
                           groups: list[str]) -> pd.DataFrame:
    """
    Where does the VMS's single text-level estimate fall relative to each
    quire-construction group's 8 independent per-text values? Reported as
    an exact rank-based (permutation) statistic appropriate for N=1 vs N=8,
    NOT as a Mann-Whitney p-value (which cannot reach significance at this
    N regardless of true effect size).
    """
    rows = []
    for g in groups:
        vals = baseline_text_level.loc[
            baseline_text_level["bpq_label"] == g, "text_level_median"
        ].values
        n = len(vals)
        if n == 0:
            continue
        rank = int((vals < vms_value).sum()) + 1  # 1-indexed rank of VMS if inserted
        # Exact permutation p-value for "VMS is an extreme value among these n+1"
        # two-sided: probability a random one of the n+1 positions is at least
        # this extreme (rank 1 and rank n+1 are the two most extreme positions)
        pos_from_edge = min(rank, (n + 1) - rank + 1)
        p_exact = 2 * pos_from_edge / (n + 1)
        p_exact = min(p_exact, 1.0)
        rows.append(dict(
            group=g, n_baseline_texts=n,
            baseline_median=np.median(vals),
            baseline_min=np.min(vals), baseline_max=np.max(vals),
            vms_value=vms_value,
            vms_rank_among_n_plus_1=rank,
            p_exact_rank_test=p_exact,
        ))
    return pd.DataFrame(rows)


def bootstrap_pairwise_median_diff(text_data: dict[str, np.ndarray], groups: list[str],
                                    n_boot: int = 20000, seed: int = 42) -> pd.DataFrame:
    """
    Text-level cluster bootstrap for EVERY pairwise group comparison, not just
    VMS vs. group. For each pair (a, b), resample texts with replacement
    independently within each group, compute the bootstrap distribution of
    median(a) - median(b), and derive a percentile CI + two-sided bootstrap
    p-value (fraction of the null-shifted resample distribution at least as
    extreme as 0 in the "wrong" direction). This directly extends the VMS
    bootstrap already in this script to the group-vs-group case, and avoids
    relying solely on the asymptotic normal approximation Mann-Whitney U uses
    at small n.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for a, b in itertools.combinations(groups, 2):
        xa, xb = text_data[a], text_data[b]
        na, nb = len(xa), len(xb)
        if na < 1 or nb < 1:
            continue
        boot_diffs = np.array([
            np.median(rng.choice(xa, size=na, replace=True))
            - np.median(rng.choice(xb, size=nb, replace=True))
            for _ in range(n_boot)
        ])
        ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])
        observed_diff = np.median(xa) - np.median(xb)
        # Two-sided bootstrap p-value: how often does the resampled
        # difference cross zero relative to the observed direction?
        p_boot = 2 * min(np.mean(boot_diffs <= 0), np.mean(boot_diffs >= 0))
        p_boot = min(p_boot, 1.0)
        rows.append(dict(
            group_a=a, group_b=b, n_a=na, n_b=nb,
            observed_median_diff=observed_diff,
            boot_ci_lo=ci_lo, boot_ci_hi=ci_hi,
            p_bootstrap=p_boot,
        ))
    df = pd.DataFrame(rows)
    n_tests = len(df)
    if n_tests:
        df["p_bootstrap_holm"] = np.nan
        order = df["p_bootstrap"].argsort().values
        p_holm = np.empty(n_tests)
        for rank, idx in enumerate(order):
            p_holm[idx] = min(df["p_bootstrap"].iloc[idx] * (n_tests - rank), 1.0)
        for i in range(1, n_tests):
            p_holm[order[i]] = max(p_holm[order[i]], p_holm[order[i - 1]])
        df["p_bootstrap_holm"] = p_holm
        df["significant"] = df["p_bootstrap_holm"] < 0.05
    return df.sort_values("p_bootstrap").reset_index(drop=True)


def bootstrap_group_comparison(baseline_text_level: pd.DataFrame, groups: list[str],
                                vms_value: float, n_boot: int = 20000,
                                seed: int = 42) -> pd.DataFrame:
    """
    Cluster bootstrap: resample the 8 source texts (with replacement) within
    each baseline group to build a sampling distribution for the group
    median, then report how often a bootstrap group median is <= the VMS
    point value. This respects "text" as the resampling unit instead of
    "run" and gives an honest, if wide, uncertainty band from only 8 texts.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for g in groups:
        vals = baseline_text_level.loc[
            baseline_text_level["bpq_label"] == g, "text_level_median"
        ].values
        n = len(vals)
        if n == 0:
            continue
        boot_medians = np.array([
            np.median(rng.choice(vals, size=n, replace=True))
            for _ in range(n_boot)
        ])
        ci_lo, ci_hi = np.percentile(boot_medians, [2.5, 97.5])
        frac_ge_vms = float(np.mean(boot_medians >= vms_value))
        rows.append(dict(
            group=g, n_texts=n, observed_median=np.median(vals),
            boot_ci_lo=ci_lo, boot_ci_hi=ci_hi,
            frac_bootstrap_medians_ge_vms=frac_ge_vms,
        ))
    return pd.DataFrame(rows)


# =============================================================================
# PLOTTING
# =============================================================================

def make_comparison_violin(baseline_raw: dict[str, np.ndarray],
                            baseline_text: dict[str, np.ndarray],
                            vms_raw: np.ndarray, vms_text_value: float,
                            groups_no_vms: list[str], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    cmap = matplotlib.colormaps.get_cmap("viridis")
    sample_points = np.linspace(0.15, 0.85, len(groups_no_vms))
    colors = [cmap(p) for p in sample_points]
    vms_color = "#e05c2a"

    # Left: original pseudoreplicated (n=576 per group, n=24 VMS)
    ax = axes[0]
    all_groups = ["VMS"] + groups_no_vms
    arrays = [vms_raw] + [baseline_raw[g] for g in groups_no_vms]
    parts = ax.violinplot(arrays, positions=range(1, len(all_groups) + 1),
                           showmedians=True, showextrema=True, widths=0.7)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(vms_color if i == 0 else colors[i - 1])
        pc.set_edgecolor("white")
        pc.set_alpha(0.75)
    ax.set_xticks(range(1, len(all_groups) + 1))
    ax.set_xticklabels([f"{g}\n(n={len(a)})" for g, a in zip(all_groups, arrays)],
                        fontsize=9)
    ax.set_title("Original: every run treated as independent\n(pseudoreplicated)",
                  fontsize=10)
    ax.set_ylabel("Multiplicative lift over random baseline")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Right: corrected, text-level (n=8 per group, n=1 VMS)
    ax = axes[1]
    arrays2 = [baseline_text[g] for g in groups_no_vms]
    parts2 = ax.violinplot(arrays2, positions=range(2, len(groups_no_vms) + 2),
                            showmedians=True, showextrema=True, widths=0.6)
    for i, pc in enumerate(parts2["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor("white")
        pc.set_alpha(0.75)
    rng = np.random.default_rng(0)
    for i, g in enumerate(groups_no_vms):
        pos = i + 2
        vals = baseline_text[g]
        jitter = rng.uniform(-0.06, 0.06, size=len(vals))
        ax.scatter(pos + jitter, vals, color=colors[i], alpha=0.8, s=28,
                   zorder=3, edgecolors="white", linewidths=0.5)
    ax.scatter([1], [vms_text_value], color=vms_color, s=90, zorder=4,
               marker="D", label="VMS (N=1 document)")
    ax.set_xticks(range(1, len(groups_no_vms) + 2))
    ax.set_xticklabels(
        ["VMS\n(N=1)"] + [f"{g}\n(N={len(baseline_text[g])} texts)" for g in groups_no_vms],
        fontsize=9,
    )
    ax.set_title("Corrected: one point per source text\n(true unit of replication)",
                  fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Comparison violin plot -> {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster-aware significance testing")
    parser.add_argument("--input", default="all_seriation_results.xlsx")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline, vms = load_full_data(Path(args.input))
    groups_no_vms = [g for g in GROUP_ORDER if g != "VMS" and g in baseline["bpq_label"].unique()]

    print("=" * 78)
    print("STEP 1 — Cluster structure check")
    print("=" * 78)
    counts = baseline.groupby(["bpq_label", "source_text"]).size()
    print(f"Baseline: {baseline['source_text'].nunique()} source texts x "
          f"{baseline['words_per_folio'].nunique()} folio lengths x "
          f"{baseline['metric'].nunique()} metrics x "
          f"{baseline['restart'].nunique()} restarts "
          f"= {counts.iloc[0]} runs per (group, text) cell")
    print(f"VMS: {vms['source_text'].nunique()} source text x "
          f"{vms['metric'].nunique()} metrics x {vms['restart'].nunique()} restarts "
          f"= {len(vms)} runs, but only 1 independent document\n")

    # ---- ORIGINAL (pseudoreplicated) analysis, for comparison ----
    print("=" * 78)
    print("STEP 2 — Original analysis (run-level, pseudoreplicated), for comparison")
    print("=" * 78)
    raw_data = {g: baseline.loc[baseline["bpq_label"] == g, VALUE_COL].values
                for g in groups_no_vms}
    raw_data["VMS"] = vms[VALUE_COL].values
    orig_df = run_pairwise_mwu(raw_data, ["VMS"] + groups_no_vms,
                                correction="bonferroni", alpha=args.alpha,
                                label="original_pseudoreplicated")
    print(orig_df[["group_a", "group_b", "n_a", "n_b", "p_raw", "p_corrected", "significant"]]
          .to_string(index=False))
    orig_df.to_csv(out_dir / "original_pseudoreplicated_tests.csv", index=False)

    # ---- CORRECTED: text-level aggregation ----
    print("\n" + "=" * 78)
    print("STEP 3 — Corrected analysis: aggregate to one value per source text")
    print("=" * 78)
    baseline_text = text_level_aggregate(baseline)
    vms_text = text_level_aggregate(vms)
    vms_value = float(vms_text["text_level_median"].iloc[0])

    text_data = {g: baseline_text.loc[baseline_text["bpq_label"] == g, "text_level_median"].values
                 for g in groups_no_vms}

    print(f"\nText-level N per group (should be 8 for baseline, 1 for VMS):")
    for g in groups_no_vms:
        print(f"  {g:12s} N={len(text_data[g])}  "
              f"median={np.median(text_data[g]):.3f}  "
              f"values={sorted(np.round(text_data[g], 2))}")
    print(f"  {'VMS':12s} N=1  value={vms_value:.3f}  "
          f"(single document; no distribution possible)\n")

    # Pairwise MWU among the 5 baseline groups only (n=8 each -- now legitimate)
    # Uses the EXACT method (network algorithm), not the normal approximation,
    # since n=8 is too small for the asymptotic version to be trustworthy.
    print("Pairwise Mann-Whitney U (exact method) among baseline quire constructions (N=8 texts each):")
    corrected_df = run_pairwise_mwu(text_data, groups_no_vms,
                                     correction="holm", alpha=args.alpha,
                                     label="corrected_text_level", method="exact")
    print(corrected_df[["group_a", "group_b", "n_a", "n_b", "statistic",
                         "p_raw", "p_corrected", "significant"]].to_string(index=False))
    corrected_df.to_csv(out_dir / "corrected_text_level_tests.csv", index=False)

    print("\nPairwise TEXT-LEVEL BOOTSTRAP (median difference, 20,000 resamples/pair), "
          "Holm-corrected:")
    boot_pairwise_df = bootstrap_pairwise_median_diff(text_data, groups_no_vms)
    print(boot_pairwise_df[["group_a", "group_b", "n_a", "n_b", "observed_median_diff",
                             "boot_ci_lo", "boot_ci_hi", "p_bootstrap",
                             "p_bootstrap_holm", "significant"]].to_string(index=False))
    boot_pairwise_df.to_csv(out_dir / "corrected_text_level_bootstrap_tests.csv", index=False)

    kw_stat, kw_p = stats.kruskal(*[text_data[g] for g in groups_no_vms])
    print(f"\nKruskal-Wallis omnibus test across all 5 groups (N=8 each): "
          f"H={kw_stat:.3f}, p={kw_p:.5f}")

    min_p = minimum_achievable_p(1, 8)
    print(f"\nNOTE: Mann-Whitney U between VMS (N=1) and any baseline group (N=8) "
          f"cannot reach p < {min_p:.3f} no matter how extreme the true effect is. "
          f"A conventional two-sample test against VMS is therefore not reported.")

    # ---- VMS vs baseline groups: rank-based + bootstrap, N=1-appropriate ----
    print("\n" + "=" * 78)
    print("STEP 4 — Where does the VMS fall, honestly reported at N=1")
    print("=" * 78)
    rank_df = vms_percentile_report(vms_value, baseline_text, groups_no_vms)
    print(rank_df.to_string(index=False))
    rank_df.to_csv(out_dir / "vms_rank_report.csv", index=False)

    print("\nCluster bootstrap (resampling the 8 source texts, 20,000 draws):")
    boot_df = bootstrap_group_comparison(baseline_text, groups_no_vms, vms_value)
    print(boot_df.to_string(index=False))
    boot_df.to_csv(out_dir / "vms_bootstrap_comparison.csv", index=False)

    # ---- Plot ----
    print("\n" + "=" * 78)
    print("STEP 5 — Plot")
    print("=" * 78)
    make_comparison_violin(raw_data, text_data, vms["mult_over_baseline"].values,
                            vms_value, groups_no_vms,
                            out_dir / "violin_comparison.svg")

    print("\nDone.")


if __name__ == "__main__":
    main()
