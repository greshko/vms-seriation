"""
figure2_quad.py -- Figure 2 as four panels (a-d), box plots throughout.

  a  35 Latin texts: per-text median CAROB (box + every text as a dot)      [violin_v2 data]
  b  VMS-like miscellany, nested quires (runs)                              [boxplot_v2 design]
  c  self-citation text, nested quires (runs); open diamonds = panel b's medians
  d  miscellany, same folia, conjoint pairs re-drawn: consecutive -> blocks of 4/8/12 -> whole source

Shared y-axis (0-22), one VMS median + bands drawn identically in every panel, one legend strip.
Needs violin_v2.py alongside.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import violin_v2 as v

VMS_COLOR, GREY = "#e05c2a", "#8a8f98"
BOX, WHIS = (25.0, 75.0), (5.0, 95.0)
QUIRES = ["Singulion", "Binion", "Ternion", "Quaternion", "Quinion"]
FS = 7.0


def stats_of(x):
    q = np.percentile(x, [WHIS[0], BOX[0], 50, BOX[1], WHIS[1]])
    return dict(whislo=q[0], q1=q[1], med=q[2], q3=q[3], whishi=q[4], fliers=[])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", type=Path, required=True)
    ap.add_argument("--vms", type=Path, required=True)
    ap.add_argument("--latin", type=Path, required=True)
    ap.add_argument("--selfcite", type=Path, required=True)
    ap.add_argument("--scribe", type=Path, required=True)
    ap.add_argument("--width", type=float, default=6.5)
    ap.add_argument("--height", type=float, default=5.9)
    ap.add_argument("--ymax", type=float, default=22.0)
    ap.add_argument("--out-dir", type=Path, default=Path("figure2_quad_out"))
    ap.add_argument("--pdf", action="store_true"); ap.add_argument("--png", action="store_true")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS - 0.8,
                         "ytick.labelsize": FS, "legend.fontsize": FS, "svg.fonttype": "none", "pdf.fonttype": 42})
    viridis = matplotlib.colormaps["viridis"]
    qcols = [viridis(p) for p in np.linspace(0.15, 0.85, 5)]

    vms = v.read_results(a.vms)
    vv = vms[v.VALUE_COL].to_numpy(float)
    vms_median, n_vms = float(np.median(vv)), len(vv)
    bands = v.vms_band_table(vv, v.VMS_BANDS_DEFAULT)

    cal = v.read_results(a.calib)
    cal["bpq_label"] = cal["bifolios_per_quire"].map(v.bpq_label)
    tl = v.text_level_aggregate(cal)
    A = [tl.loc[tl.bpq_label == g, "text_level_median"].to_numpy(float) for g in QUIRES]
    lat = pd.read_csv(a.latin)
    B = [lat.loc[lat.bifolios_per_quire == b, "carob"].to_numpy(float) for b in range(1, 6)]
    sc = pd.read_csv(a.selfcite)
    C = [sc.loc[sc.pairing == f"nested_bpq{b}", "carob"].to_numpy(float) for b in range(1, 6)]
    sr = pd.read_csv(a.scribe)
    d_arms = [("nested_bpq1", "consec-\nutive"), ("blk4", "block\nof 4"), ("blk8", "block\nof 8"),
              ("blk12", "block\nof 12"), ("src_winf", "whole\nsource")]
    D = [sr.loc[sr.pairing == p, "carob"].to_numpy(float) for p, _ in d_arms]

    fig = plt.figure(figsize=(a.width, a.height))
    gs = GridSpec(2, 2, figure=fig, left=0.095, right=0.99, top=0.835, bottom=0.09, wspace=0.30, hspace=0.50)
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]

    def reference(ax):
        ax.axhline(1.0, color="#888", ls=":", lw=1.0, zorder=1)
        for (c, lo, hi), al in zip(bands, np.linspace(0.13, 0.30, len(bands))):
            ax.axhspan(lo, hi, facecolor=VMS_COLOR, alpha=float(al), lw=0, zorder=0.6)
        ax.axhline(vms_median, color=VMS_COLOR, ls="--", lw=1.3, alpha=0.9, zorder=2)

    def draw(ax, groups, colors, hatch=None, dots=False):
        for i, (vals, col) in enumerate(zip(groups, colors), start=1):
            bp = ax.bxp([stats_of(vals)], positions=[i], widths=0.6, patch_artist=True, showfliers=False,
                        boxprops=dict(linewidth=1.0, edgecolor="#333"), medianprops=dict(color="#111", linewidth=1.8),
                        whiskerprops=dict(color="#333", linewidth=1.0), capprops=dict(color="#333", linewidth=1.0))
            p = bp["boxes"][0]
            p.set_facecolor(matplotlib.colors.to_rgba(col, 0.55 if hatch else 0.75)); p.set_zorder(1.6)
            if hatch:
                p.set_hatch(hatch); p.set_edgecolor("#333")
            for k in ("whiskers", "caps"):
                for ln in bp[k]:
                    ln.set_zorder(1.7)
            bp["medians"][0].set_zorder(1.9)
        if dots:
            rng = np.random.default_rng(0)
            for i, vals in enumerate(groups, start=1):
                ax.scatter(i + rng.uniform(-0.17, 0.17, len(vals)), vals, s=8, color="#222", alpha=0.55,
                           zorder=3, edgecolors="white", linewidths=0.3)

    draw(axs[0], A, qcols, dots=True)
    draw(axs[1], B, qcols)
    draw(axs[2], C, qcols, hatch="////")
    draw(axs[3], D, [GREY] * 5)
    axs[2].scatter(range(1, 6), [np.median(x) for x in B], marker="D", s=16, facecolor="none",
                   edgecolor="k", linewidth=0.9, zorder=4)

    titles = [("a", "35 Latin texts", "one point per text (median of its runs)"),
              ("b", "VMS-like miscellany", "150 runs per box"),
              ("c", "Self-citation text", "960 runs of one text; \u25c7 = median in b"),
              ("d", "Miscellany, pairs re-drawn", "120 runs per box; same five source texts as b")]
    for ax, (lab, t, sub) in zip(axs, titles):
        reference(ax)
        ax.set_ylim(0, a.ymax); ax.set_xlim(0.4, 5.6)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.set_title(f"{t}", fontsize=FS + 0.5, loc="left", pad=11, fontweight="bold")
        ax.text(0.0, 1.015, sub, transform=ax.transAxes, fontsize=FS - 0.7, color="#555", va="bottom")
        ax.text(-0.19, 1.2, lab, transform=ax.transAxes, fontsize=11, fontweight="bold",
                ha="right", va="bottom")
    for ax in axs[:3]:
        ax.set_xticks(range(1, 6)); ax.set_xticklabels(QUIRES, fontsize=FS - 1.1)
    axs[3].set_xticks(range(1, 6)); axs[3].set_xticklabels([l for _, l in d_arms], fontsize=FS - 0.8)
    for ax, lab in zip(axs, ["CAROB (median per text)", "CAROB (individual runs)",
                             "CAROB (individual runs)", "CAROB (individual runs)"]):
        ax.set_ylabel(lab)

    # legend strip
    hs = [Line2D([0], [0], color=VMS_COLOR, ls="--", lw=1.3, label=f"VMS median CAROB ({vms_median:.2f}, {n_vms} runs)"),
          matplotlib.patches.Patch(facecolor=VMS_COLOR, alpha=0.30, label=f"VMS central 50% of runs ({bands[1][1]:.2f}\u2013{bands[1][2]:.2f})"),
          matplotlib.patches.Patch(facecolor=VMS_COLOR, alpha=0.13, label=f"VMS central 90% of runs ({bands[0][1]:.2f}\u2013{bands[0][2]:.2f})"),
          Line2D([0], [0], color="#888", ls=":", lw=1.0, label="random ordering (CAROB = 1)")]
    fig.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.54, 1.0), ncol=2, frameon=False, fontsize=FS,
               handlelength=1.8, columnspacing=1.6)
    fig.text(0.54, 0.012, f"box = {BOX[0]:g}th\u2013{BOX[1]:g}th percentile, whiskers = {WHIS[0]:g}th\u2013{WHIS[1]:g}th, line = median",
             ha="center", va="bottom", fontsize=FS - 0.5, color="#555")
    out = a.out_dir / "figure2_quad"
    fig.savefig(out.with_suffix(".svg"), format="svg")
    if a.pdf: fig.savefig(out.with_suffix(".pdf"))
    if a.png: fig.savefig(out.with_suffix(".png"), dpi=220)
    plt.close(fig); print("wrote", out.with_suffix(".svg"))


if __name__ == "__main__":
    main()
