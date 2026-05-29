"""
violin_significance.py
======================
Violin plots with pairwise significance tests for the bifolio adjacency
multiplicative lift data.

Reads a two-column TSV or CSV with columns:
    quire_construction  group label (numeric 1-5 or named; see LABEL_MAP)
    mult_over_baseline  numeric value

Numeric bpq values (1-5) are automatically mapped to quire construction
names (Singulion, Binion, Trinion, Quaternion, Quinion) before plotting.
Old files using bpq_label as the column name are accepted transparently.

Produces:
    violin_significance.svg   publication-ready violin plot with significance
                              brackets and p-value annotations
    pairwise_tests.csv        full pairwise test results table

Usage
-----
    python violin_significance.py --input data.csv

    # Override defaults:
    python violin_significance.py --input data.csv \\
        --output-dir outputs \\
        --test mwu \\
        --correction bonferroni \\
        --alpha 0.05 \\
        --value-col mult_over_baseline \\
        --group-col quire_construction

Configuration
-------------
Edit DEFAULTS below or pass command-line arguments.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats

# =============================================================================
# DEFAULTS  (override via command-line arguments)
# =============================================================================

# Canonical display order for quire construction groups.
# Numeric bpq values (1-5) are mapped to these names before plotting.
LABEL_MAP = {
    "1": "Singulion",
    "2": "Binion",
    "3": "Trinion",
    "4": "Quaternion",
    "5": "Quinion",
}

# Desired display order — VMS always first.
GROUP_ORDER = ["VMS", "Singulion", "Binion", "Trinion", "Quaternion", "Quinion"]

DEFAULTS = dict(
    input        = "data.csv",
    output_dir   = ".",
    value_col    = "multiplicative_lift_over_random_baseline",
    group_col    = "quire_construction",
    test         = "mwu",          # "mwu" (Mann-Whitney U) or "ttest"
    correction   = "bonferroni",   # "bonferroni", "holm", or "none"
    alpha        = 0.05,
    fig_width    = 11.0,
    fig_height   = 7.0,
    palette      = "viridis",      # matplotlib colormap; VMS gets vms_color
    show_points  = True,           # overlay individual data points
    point_alpha  = 0.35,
    point_size   = 18,
    vms_color    = "#e05c2a",      # distinct color for VMS
    show_title    = False,         # show title on the figure
    show_brackets = False,         # draw significance brackets on the figure
    bracket_base_y_gap = 0.04,     # fraction of y-range between stacked brackets
)

# =============================================================================
# STATISTICAL HELPERS
# =============================================================================

def run_pairwise_tests(
    data:       dict[str, np.ndarray],
    groups:     list[str],
    test:       str,
    correction: str,
    alpha:      float,
) -> pd.DataFrame:
    """
    Run all pairwise tests between groups.

    Parameters
    ----------
    data       : dict label -> array of values
    groups     : ordered list of group labels
    test       : "mwu" or "ttest"
    correction : "bonferroni", "holm", or "none"
    alpha      : significance threshold (after correction)

    Returns
    -------
    DataFrame with columns:
        group_a, group_b, n_a, n_b,
        statistic, p_raw, p_corrected, significant,
        stars, effect_size_r   (rank-biserial r for MWU, Cohen's d for t-test)
    """
    pairs = list(itertools.combinations(groups, 2))
    rows  = []

    for a, b in pairs:
        xa, xb = data[a], data[b]
        if test == "mwu":
            stat, p = stats.mannwhitneyu(xa, xb, alternative="two-sided")
            # Rank-biserial correlation as effect size
            n1, n2 = len(xa), len(xb)
            r = 1 - (2 * stat) / (n1 * n2)
        else:  # ttest
            stat, p = stats.ttest_ind(xa, xb, equal_var=False)
            # Cohen's d
            pooled_sd = np.sqrt(
                (np.std(xa, ddof=1) ** 2 + np.std(xb, ddof=1) ** 2) / 2
            )
            r = (np.mean(xa) - np.mean(xb)) / pooled_sd if pooled_sd > 0 else np.nan

        rows.append({
            "group_a":       str(a),
            "group_b":       str(b),
            "n_a":           len(xa),
            "n_b":           len(xb),
            "statistic":     round(float(stat), 4),
            "p_raw":         float(p),
            "effect_size":   round(float(r), 4),
        })

    df = pd.DataFrame(rows)
    n_tests = len(df)

    # Multiple comparison correction
    if correction == "bonferroni":
        df["p_corrected"] = np.minimum(df["p_raw"] * n_tests, 1.0)
    elif correction == "holm":
        sorted_idx = df["p_raw"].argsort().values
        p_holm     = np.empty(n_tests)
        for rank, idx in enumerate(sorted_idx):
            p_holm[idx] = min(df["p_raw"].iloc[idx] * (n_tests - rank), 1.0)
        # Enforce monotonicity
        for i in range(1, n_tests):
            p_holm[sorted_idx[i]] = max(
                p_holm[sorted_idx[i]], p_holm[sorted_idx[i - 1]]
            )
        df["p_corrected"] = p_holm
    else:
        df["p_corrected"] = df["p_raw"]

    df["significant"] = df["p_corrected"] < alpha

    def stars(p):
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return "ns"

    df["stars"] = df["p_corrected"].apply(stars)

    return df.sort_values("p_corrected").reset_index(drop=True)


# =============================================================================
# VIOLIN PLOT
# =============================================================================

def make_violin(
    data:        dict[str, np.ndarray],
    groups:      list[str],
    test_df:     pd.DataFrame,
    cfg:         dict,
    out_path:    Path,
    value_col:   str,
    group_col:   str,
    test_name:   str,
    correction:  str,
    alpha:       float,
) -> None:
    """Draw violin plot with significance brackets and save as SVG."""

    n_groups = len(groups)
    fig, ax  = plt.subplots(figsize=(cfg["fig_width"], cfg["fig_height"]))

    # ── Colors ────────────────────────────────────────────────────────────────
    # Sample the colormap evenly across [0.15, 0.85] so all non-VMS groups get
    # visually distinct colors without hitting the very dark/light extremes.
    # VMS always gets its own fixed accent color regardless of palette.
    non_vms_count = sum(1 for g in groups if "VMS" not in str(g))
    sample_points = np.linspace(0.15, 0.85, max(non_vms_count, 1))
    cmap          = matplotlib.colormaps.get_cmap(cfg["palette"])
    color_iter    = iter(sample_points)
    colors        = [
        cfg["vms_color"] if "VMS" in str(g) else cmap(next(color_iter))
        for g in groups
    ]

    # ── Violins ───────────────────────────────────────────────────────────────
    positions = list(range(1, n_groups + 1))
    arrays    = [data[g] for g in groups]

    parts = ax.violinplot(
        arrays,
        positions  = positions,
        showmedians= True,
        showextrema= True,
        widths      = 0.7,
    )

    for i, (pc, color) in enumerate(zip(parts["bodies"], colors)):
        pc.set_facecolor(color)
        pc.set_edgecolor("white")
        pc.set_alpha(0.75)

    for part_name in ("cmedians", "cmins", "cmaxes", "cbars"):
        if part_name in parts:
            parts[part_name].set_color("#333333")
            parts[part_name].set_linewidth(1.2)

    # ── Individual points ─────────────────────────────────────────────────────
    if cfg["show_points"]:
        rng = np.random.default_rng(42)
        for i, (g, pos) in enumerate(zip(groups, positions)):
            vals   = data[g]
            jitter = rng.uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(
                pos + jitter, vals,
                color  = colors[i],
                alpha  = cfg["point_alpha"],
                s      = cfg["point_size"],
                zorder = 3,
                linewidths = 0,
            )

    # ── Axis labels & ticks ───────────────────────────────────────────────────
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"{g}\n(n={len(data[g])})" for g in groups],
        fontsize = 10,
    )
    ax.set_ylabel(value_col.replace("_", " ").title(), fontsize=11)
    ax.set_xlabel(group_col.replace("_", " ").title(), fontsize=11)
    ax.tick_params(axis="y", labelsize=9)

    test_label = "Mann-Whitney U" if test_name == "mwu" else "Welch's t-test"
    corr_label = correction.capitalize() if correction != "none" else "uncorrected"
    if cfg.get("show_title"):
        ax.set_title(
            f"Multiplicative lift over random baseline by quire construction\n"
            f"({test_label}, {corr_label} correction, α={alpha})",
            fontsize = 11,
            pad      = 10,
        )

    # ── Significance brackets ─────────────────────────────────────────────────
    sig_pairs = test_df[test_df["significant"]].copy() if cfg.get("show_brackets") else pd.DataFrame()

    if not sig_pairs.empty:
        y_max   = max(v.max() for v in arrays)
        y_range = y_max - min(v.min() for v in arrays)
        gap     = y_range * cfg["bracket_base_y_gap"]
        y_base  = y_max + gap

        # Sort by span width so shorter brackets sit lower
        def span(row):
            i = groups.index(row["group_a"])
            j = groups.index(row["group_b"])
            return abs(j - i)

        sig_pairs = sig_pairs.copy()
        sig_pairs["_span"] = sig_pairs.apply(span, axis=1)
        sig_pairs = sig_pairs.sort_values("_span").reset_index(drop=True)

        # Track y-levels to avoid overlap
        used_y: dict[tuple, float] = {}

        for _, row in sig_pairs.iterrows():
            if row["group_a"] not in groups or row["group_b"] not in groups:
                continue
            i = groups.index(row["group_a"])
            j = groups.index(row["group_b"])
            x1, x2 = positions[i], positions[j]

            # Find a y level that doesn't overlap with existing brackets
            y_level = y_base
            while any(
                abs(y_level - v) < gap * 0.9
                and not (max(x1, x2) < min(k[0], k[1]) or min(x1, x2) > max(k[0], k[1]))
                for k, v in used_y.items()
            ):
                y_level += gap

            used_y[(x1, x2)] = y_level

            h = gap * 0.25
            ax.plot(
                [x1, x1, x2, x2],
                [y_level - h, y_level, y_level, y_level - h],
                lw    = 1.2,
                color = "#333333",
            )
            label = row["stars"]
            ax.text(
                (x1 + x2) / 2, y_level + gap * 0.05,
                label,
                ha       = "center",
                va       = "bottom",
                fontsize = 9,
                color    = "#333333",
            )

        # Extend y-axis to accommodate brackets
        ax.set_ylim(bottom=ax.get_ylim()[0],
                    top=max(used_y.values()) + gap * 1.5 if used_y else y_max + gap)

    # ── Legend: significance key (only shown when brackets are visible) ────────
    if cfg.get("show_brackets"):
        legend_elements = [
            mpatches.Patch(facecolor="none", edgecolor="none", label="* p<0.05"),
            mpatches.Patch(facecolor="none", edgecolor="none", label="** p<0.01"),
            mpatches.Patch(facecolor="none", edgecolor="none", label="*** p<0.001"),
            mpatches.Patch(facecolor="none", edgecolor="none", label="ns  not significant"),
        ]
        ax.legend(
            handles   = legend_elements,
            loc       = "upper right",
            fontsize  = 8,
            framealpha= 0.7,
            title     = f"{corr_label}-corrected p",
            title_fontsize = 8,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Violin plot -> {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Violin plots + pairwise significance tests"
    )
    parser.add_argument("--input",       default=DEFAULTS["input"],
                        help="Input TSV or CSV file")
    parser.add_argument("--output-dir",  default=DEFAULTS["output_dir"],
                        help="Directory for output files")
    parser.add_argument("--value-col",   default=DEFAULTS["value_col"],
                        help="Name of the numeric column")
    parser.add_argument("--group-col",   default=DEFAULTS["group_col"],
                        help="Name of the group label column")
    parser.add_argument("--test",        default=DEFAULTS["test"],
                        choices=["mwu", "ttest"],
                        help="Pairwise test: mwu or ttest")
    parser.add_argument("--correction",  default=DEFAULTS["correction"],
                        choices=["bonferroni", "holm", "none"],
                        help="Multiple comparison correction")
    parser.add_argument("--alpha",       default=DEFAULTS["alpha"], type=float,
                        help="Significance threshold")
    parser.add_argument("--show-title",    action="store_true",
                        help="Show title on the figure (off by default)")
    parser.add_argument("--show-brackets", action="store_true",
                        help="Draw significance brackets on the figure "
                             "(off by default; see pairwise_tests.csv for full table)")
    parser.add_argument("--no-points",   action="store_true",
                        help="Suppress individual data points")
    parser.add_argument("--palette",     default=DEFAULTS["palette"],
                        help="Color palette: 'default' or matplotlib colormap name")
    parser.add_argument("--fig-width",   default=DEFAULTS["fig_width"], type=float)
    parser.add_argument("--fig-height",  default=DEFAULTS["fig_height"], type=float)

    args = parser.parse_args()

    cfg = dict(DEFAULTS)
    cfg["fig_width"]   = args.fig_width
    cfg["fig_height"]  = args.fig_height
    cfg["show_points"]   = not args.no_points
    cfg["show_title"]    = args.show_title
    cfg["show_brackets"] = args.show_brackets
    cfg["palette"]       = args.palette

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    in_path = Path(args.input)

    # Auto-detect separator by sniffing the first non-empty line.
    # Handles CSV, TSV, semicolon-delimited, and pipe-delimited files.
    # Falls back to extension-based detection if sniffing fails.
    import csv as _csv
    try:
        with open(in_path, encoding="utf-8-sig") as _fh:
            _sample = ""
            for _line in _fh:
                _stripped = _line.strip()
                if _stripped:
                    _sample = _stripped
                    break
        _dialect = _csv.Sniffer().sniff(_sample, delimiters=",\t;|")
        sep = _dialect.delimiter
    except Exception:
        sep = "\t" if in_path.suffix.lower() in (".tsv", ".txt") else ","

    df = pd.read_csv(in_path, sep=sep, encoding="utf-8-sig")

    # ── Column aliasing: accept bpq_label or quire_construction ─────────────
    if args.group_col not in df.columns and "bpq_label" in df.columns:
        df = df.rename(columns={"bpq_label": args.group_col})

    missing = {args.value_col, args.group_col} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in input: {missing}\n"
                         f"Available: {list(df.columns)}")

    df[args.value_col] = pd.to_numeric(df[args.value_col], errors="coerce")
    df = df.dropna(subset=[args.value_col, args.group_col])

    # ── Label mapping: numeric bpq values -> quire construction names ─────────
    df[args.group_col] = df[args.group_col].astype(str).map(
        lambda x: LABEL_MAP.get(x, x)
    )

    # ── Group ordering: follow GROUP_ORDER; append unexpected labels at end ───
    present = set(df[args.group_col].unique())
    groups  = [g for g in GROUP_ORDER if g in present]
    extras  = sorted(present - set(groups))
    groups  = groups + extras
    data   = {g: df.loc[df[args.group_col] == g, args.value_col].values
              for g in groups}

    print(f"\nGroups: {groups}")
    for g in groups:
        arr = data[g]
        print(f"  {g:20s}  n={len(arr):4d}  "
              f"mean={np.mean(arr):.3f}  median={np.median(arr):.3f}  "
              f"sd={np.std(arr, ddof=1):.3f}")

    # ── Pairwise tests ────────────────────────────────────────────────────────
    print(f"\nRunning pairwise {args.test.upper()} tests "
          f"({args.correction} correction, α={args.alpha}) ...")

    test_df  = run_pairwise_tests(
        data, groups, args.test, args.correction, args.alpha
    )

    csv_path = out_dir / "pairwise_tests.csv"
    test_df.to_csv(csv_path, index=False)
    print(f"  Pairwise tests -> {csv_path}")

    sig = test_df[test_df["significant"]]
    print(f"\n  {len(sig)}/{len(test_df)} pairs significant at α={args.alpha} "
          f"({args.correction}-corrected):")
    for _, row in sig.iterrows():
        print(f"    {row['group_a']:20s} vs {row['group_b']:20s}  "
              f"p_corr={row['p_corrected']:.4f}  {row['stars']}")

    # ── Violin plot ───────────────────────────────────────────────────────────
    svg_path = out_dir / "violin_significance.svg"
    make_violin(
        data       = data,
        groups     = groups,
        test_df    = test_df,
        cfg        = cfg,
        out_path   = svg_path,
        value_col  = args.value_col,
        group_col  = args.group_col,
        test_name  = args.test,
        correction = args.correction,
        alpha      = args.alpha,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()