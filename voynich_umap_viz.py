"""
voynich_umap_viz.py
===================
NMF topic modeling + UMAP dimensionality reduction + seriation visualization
for the Voynich Manuscript at the bifolio level.

For each topic count K in K_VALUES:
  1. Fit NMF on bifolio-level token counts (bifolio_global as doc_id)
  2. Project the K-dimensional topic vectors to 2D via UMAP
  3. Produce ONE two-panel figure per K, sized for a single-column/full-width
     paper figure:
         (a) seriation path following the original (bound) bifolio ordering
         (b) seriation path following the consensus seriation
     Both panels share the same UMAP embedding, axes, and colorbar, so the
     same bifolium sits at the same position in each panel.
  4. Write UMAP coordinates to umap_coords_K<K>.csv

Outputs (in OUTPUT_DIR):
    figure3_K<K>.svg / .png      two-panel figure (a | b)
    umap_coords_K<K>.csv         UMAP coordinates + NMF topic weights

Usage
-----
    python voynich_umap_viz.py

Requires: numpy, pandas, scikit-learn, umap-learn, matplotlib
Optional: adjustText  (non-overlapping bifolium labels; falls back to simple
          offset labels if it is not installed)
Install:  pip install umap-learn matplotlib adjustText
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend; file output only
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize

try:
    from adjustText import adjust_text
except ImportError:          # labels fall back to a fixed offset above each node
    adjust_text = None

# =============================================================================
# CONFIGURATION
# =============================================================================

FOLIATION_CSV   = Path("voynich/vms_rf1b_folio_bifolio.csv")
CONSENSUS_CSV   = Path("outputs_final/consensus_seriation/consensus_vs_original.csv")
OUTPUT_DIR      = Path("outputs/voynich_umap")

K_VALUES        = [8, 12, 16, 20]   # topic counts to visualize

NMF_SEED        = 42
UMAP_SEED       = 42
UMAP_N_NEIGHBORS = 10    # lower = more local structure
UMAP_MIN_DIST    = 0.1

# ---- Output -----------------------------------------------------------------
FORMATS         = ("svg", "png")   # file types to write
PNG_DPI         = 600

# ---- Figure geometry --------------------------------------------------------
# Designed at the size it will be printed. 6.3 in is roughly the full text width
# of the paper's page, so fonts below appear at their true point sizes when the
# figure is included at \textwidth (do not rescale it much in the document).
FIG_SIZE        = (6.3, 3.6)       # inches (width, height)

# ---- Typography (points, at final size) -------------------------------------
FONT_PANEL_LETTER = 11    # bold "a" / "b"
FONT_TITLE        = 8.5   # panel titles
FONT_AXIS_LABEL   = 8     # "UMAP 1", "UMAP 2", colorbar label
FONT_TICK         = 7     # axis and colorbar tick labels
FONT_LEGEND       = 7.5
FONT_NODE_LABEL   = 6.0   # bifolium numbers next to nodes
SHOW_NODE_LABELS  = True  # set False for an unlabeled version
LEADER_MIN_PT     = 5     # draw a leader line if a label sits farther than this (pt) from its node

# ---- Marks ------------------------------------------------------------------
NODE_SIZE       = 12
NODE_COLOR      = "#333333"
NODE_ALPHA      = 0.9
LINE_ALPHA      = 0.85
LINE_WIDTH      = 1.1
COLORMAP        = "plasma"         # gradient colormap for seriation path
START_MARKER_SIZE = 75             # star marking bifolium 1
END_MARKER_SIZE   = 42             # square marking final bifolium
AXIS_PAD_FRAC   = 0.07             # padding around data extent

# =============================================================================
# DATA HELPERS
# =============================================================================

def load_bifolio_tokens(csv_path: Path) -> pd.DataFrame:
    """
    Load foliation CSV, remap bifolio_global -> doc_id, return token-level df.
    Drops the original doc_id (folio number); each row is (bifolio, token).
    """
    raw = pd.read_csv(csv_path, encoding="utf-8-sig")
    raw["doc_id"]         = raw["doc_id"].astype(int)
    raw["bifolio_global"] = raw["bifolio_global"].astype(int)
    df = raw.copy()
    df["doc_id"] = df["bifolio_global"]
    return df[["doc_id", "token"]]


def prune_vocabulary(counts: pd.DataFrame, n_max: int = 3000) -> pd.DataFrame:
    n_docs   = counts.shape[0]
    doc_freq = (counts > 0).sum(axis=0)
    mask     = (doc_freq >= 2) & (doc_freq <= 0.8 * n_docs)
    pruned   = counts.loc[:, mask]
    if pruned.shape[1] > n_max:
        top    = counts.sum(axis=0)[mask].sort_values(ascending=False).head(n_max).index
        pruned = pruned[top]
    return pruned


def fit_nmf_best(X, K: int, n_seeds: int = 5, base_seed: int = 42):
    """Fit NMF with n_seeds, return W for the seed with lowest reconstruction error."""
    best_W, best_err = None, float("inf")
    for s in range(n_seeds):
        nmf = NMF(n_components=K, init="nndsvda",
                  random_state=base_seed + s, max_iter=3000)
        W   = nmf.fit_transform(X)
        if nmf.reconstruction_err_ < best_err:
            best_err = nmf.reconstruction_err_
            best_W   = normalize(W, norm="l1", axis=1)
    return best_W


def load_orderings(consensus_csv: Path) -> tuple[list[int], list[int]]:
    """
    Load consensus_vs_original.csv.
    Returns (original_order, consensus_order) as lists of bifolio_global values,
    in rank order.
    """
    df = pd.read_csv(consensus_csv, encoding="utf-8-sig")
    df = df.sort_values("rank")
    return df["original"].tolist(), df["consensus"].tolist()


# =============================================================================
# FIGURE
# =============================================================================

def _draw_panel(
    ax,
    xy:          np.ndarray,          # (n_bifolios, 2) UMAP coordinates
    bg_to_row:   dict[int, int],      # bifolio_global -> row in xy
    ordering:    list[int],           # bifolio_global values in sequence order
    cmap,
    norm,
) -> list[int]:
    """Draw nodes, gradient seriation path, and start/end markers on one axes."""
    valid = [bg for bg in ordering if bg in bg_to_row]
    if len(valid) < 2:
        print("  [warn] Fewer than two valid bifolios in ordering; panel left empty.")
        return valid

    xs, ys = xy[:, 0], xy[:, 1]
    path   = xy[[bg_to_row[bg] for bg in valid]]

    # Gradient path as a single LineCollection; each segment is colored by the
    # midpoint of the two ranks it connects (1.5, 2.5, ...).
    segments = np.stack([path[:-1], path[1:]], axis=1)
    lc = LineCollection(
        segments, cmap=cmap, norm=norm,
        linewidths=LINE_WIDTH, alpha=LINE_ALPHA,
        capstyle="round", joinstyle="round", zorder=2,
    )
    lc.set_array(np.arange(1, len(valid)) + 0.5)
    ax.add_collection(lc)

    # All bifolia as nodes
    ax.scatter(xs, ys, s=NODE_SIZE, color=NODE_COLOR, alpha=NODE_ALPHA,
               zorder=3, linewidths=0)

    # Start and end markers
    ax.scatter(*path[0], s=START_MARKER_SIZE, color=cmap(0.0), marker="*",
               edgecolors="white", linewidths=0.5, zorder=5)
    ax.scatter(*path[-1], s=END_MARKER_SIZE, color=cmap(1.0), marker="s",
               edgecolors="white", linewidths=0.5, zorder=5)
    return valid


def _label_nodes(axes, xy: np.ndarray, bifolio_ids: list[int]) -> None:
    """
    Number each node in every panel.

    Label positions are optimized once (adjustText, on the first panel) and then
    reused in the other panels, so a given bifolium's label sits in the same
    place in (a) and (b). Falls back to a fixed offset above each node if
    adjustText is not installed.
    """
    halo = [pe.withStroke(linewidth=1.6, foreground="white")]
    leader = dict(arrowstyle="-", color="#777777", lw=0.3, shrinkA=0, shrinkB=1)
    xs, ys = xy[:, 0], xy[:, 1]

    if adjust_text is None:
        for ax in axes:
            for bg, x, y in zip(bifolio_ids, xs, ys):
                ax.annotate(
                    str(bg), (x, y), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=FONT_NODE_LABEL,
                    color="#222222", zorder=6, path_effects=halo,
                )
        return

    ax0 = axes[0]
    texts = [
        ax0.text(x, y, str(bg), ha="center", va="center",
                 fontsize=FONT_NODE_LABEL, color="#222222",
                 zorder=6, path_effects=halo)
        for bg, x, y in zip(bifolio_ids, xs, ys)
    ]
    adjust_text(
        texts, x=xs, y=ys, ax=ax0,
        expand=(1.15, 1.35),
        force_text=(0.3, 0.5), force_static=(0.5, 0.8),
        min_arrow_len=LEADER_MIN_PT,
        arrowprops=leader,
    )
    final_pos = [t.get_position() for t in texts]

    # Reuse the optimized positions in the remaining panels.
    for ax in axes[1:]:
        for bg, (x, y), (tx, ty) in zip(bifolio_ids, xy, final_pos):
            d_px = ax.transData.transform((tx, ty)) - ax.transData.transform((x, y))
            d_pt = float(np.hypot(*d_px)) * 72.0 / ax.figure.dpi
            ax.annotate(
                str(bg), xy=(x, y), xytext=(tx, ty), textcoords="data",
                ha="center", va="center", fontsize=FONT_NODE_LABEL,
                color="#222222", zorder=6, path_effects=halo,
                arrowprops=leader if d_pt > LEADER_MIN_PT else None,
                annotation_clip=False,
            )


def make_figure(
    embedding:       np.ndarray,       # (n_bifolios, 2)
    bifolio_ids:     list[int],        # parallel to rows of embedding
    original_order:  list[int],
    consensus_order: list[int],
    K:               int,
    out_stem:        Path,             # path without extension
) -> None:
    """
    Two-panel figure: (a) original ordering | (b) consensus seriation.
    Axes, axis labels, and colorbar are shared between panels.
    """
    plt.rcParams.update({
        "font.size":        FONT_TICK,
        "axes.linewidth":   0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size":  2.5,
        "ytick.major.size":  2.5,
        "xtick.major.pad":   2,
        "ytick.major.pad":   2,
    })

    bg_to_row = {bg: i for i, bg in enumerate(bifolio_ids)}
    cmap = matplotlib.colormaps[COLORMAP]
    norm = mcolors.Normalize(vmin=1, vmax=len(original_order))

    fig, axes = plt.subplots(
        1, 2, figsize=FIG_SIZE, sharex=True, sharey=True,
        layout="constrained",
    )
    fig.get_layout_engine().set(w_pad=0.04, h_pad=0.03, wspace=0.04)

    # Shared axis limits (identical node positions in both panels)
    lo, hi = embedding.min(axis=0), embedding.max(axis=0)
    pad    = (hi - lo) * AXIS_PAD_FRAC
    axes[0].set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    axes[0].set_ylim(lo[1] - pad[1], hi[1] + pad[1])

    panels = [
        ("a", "Original ordering",    original_order),
        ("b", "Consensus seriation",  consensus_order),
    ]
    for ax, (letter, title, order) in zip(axes, panels):
        _draw_panel(ax, embedding, bg_to_row, order, cmap, norm)

        ax.set_title(title, loc="left", fontsize=FONT_TITLE,
                     fontweight="bold", pad=5)
        ax.annotate(letter, xy=(0, 1), xycoords="axes fraction",
                    xytext=(-3, 5), textcoords="offset points",
                    ha="right", va="bottom",
                    fontsize=FONT_PANEL_LETTER, fontweight="bold")

        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.tick_params(labelsize=FONT_TICK)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    # Consolidated axis labels: one x label centered under both panels, one
    # y label at the far left. (Inner y tick labels are hidden by sharey.)
    fig.supxlabel("UMAP 1", fontsize=FONT_AXIS_LABEL)
    fig.supylabel("UMAP 2", fontsize=FONT_AXIS_LABEL)

    # One shared colorbar for both panels
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    n = len(original_order)
    cbar = fig.colorbar(sm, ax=axes, location="right", fraction=0.035,
                        pad=0.015, aspect=22)
    cbar.set_label("Seriation rank", fontsize=FONT_AXIS_LABEL, labelpad=3)
    cbar.set_ticks([1, 10, 20, 30, 40, n])
    cbar.ax.tick_params(labelsize=FONT_TICK, width=0.6, length=2.5)
    cbar.outline.set_linewidth(0.6)

    # One shared legend for the start/end markers (above the panels)
    handles = [
        Line2D([], [], marker="*", linestyle="none", markersize=8,
               markerfacecolor=cmap(0.0), markeredgecolor="white",
               markeredgewidth=0.5, label="Start (bifolium 1)"),
        Line2D([], [], marker="s", linestyle="none", markersize=5,
               markerfacecolor=cmap(1.0), markeredgecolor="white",
               markeredgewidth=0.5, label="End"),
    ]
    fig.legend(handles=handles, loc="outside upper right", ncol=2,
               frameon=False, fontsize=FONT_LEGEND, handletextpad=0.2,
               columnspacing=1.2, borderaxespad=0.0)

    # Number the nodes only after layout is final, so label placement is
    # computed against the real axes size.
    if SHOW_NODE_LABELS:
        fig.canvas.draw()
        _label_nodes(axes, embedding, bifolio_ids)

    for ext in FORMATS:
        path = out_stem.with_suffix(f".{ext}")
        kwargs = {"dpi": PNG_DPI} if ext == "png" else {}
        fig.savefig(path, format=ext, facecolor="white", **kwargs)
        print(f"    Saved: {path.name}")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Voynich bifolio UMAP visualization")
    print(f"  Foliation : {FOLIATION_CSV}")
    print(f"  Consensus : {CONSENSUS_CSV}")
    print(f"  K values  : {K_VALUES}")
    print("=" * 60)

    # --- Load tokens and build count matrix ---
    print("\nLoading tokens and building count matrix ...")
    df = load_bifolio_tokens(FOLIATION_CSV)

    counts_full = (
        df.groupby(["doc_id", "token"])
          .size()
          .unstack(fill_value=0)
          .sort_index()
    )
    counts = prune_vocabulary(counts_full)
    bifolio_ids = counts.index.tolist()
    n_bifolios  = len(bifolio_ids)

    print(f"  Bifolios after pruning : {n_bifolios}")
    print(f"  Vocabulary size        : {counts.shape[1]}")

    # TF-IDF
    tfidf = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True)
    X     = tfidf.fit_transform(counts.values)

    # --- Load orderings ---
    original_order, consensus_order = load_orderings(CONSENSUS_CSV)
    print(f"  Original ordering  : {len(original_order)} bifolios")
    print(f"  Consensus ordering : {len(consensus_order)} bifolios")

    # --- Per-K loop ---
    for K in K_VALUES:
        print(f"\n--- K = {K} ---")

        # NMF
        print(f"  Fitting NMF (K={K}) ...")
        W = fit_nmf_best(X, K, n_seeds=5, base_seed=NMF_SEED)
        # W shape: (n_bifolios, K)

        # UMAP
        print(f"  Running UMAP ...")
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=UMAP_N_NEIGHBORS,
            min_dist=UMAP_MIN_DIST,
            random_state=UMAP_SEED,
            metric="hellinger",   # appropriate for probability distributions
        )
        embedding = reducer.fit_transform(W)   # (n_bifolios, 2)

        # Save UMAP coordinates
        coords_df = pd.DataFrame({
            "bifolio_global": bifolio_ids,
            "umap1":          embedding[:, 0].round(6),
            "umap2":          embedding[:, 1].round(6),
        })
        # Also attach NMF topic weights for reference
        for t in range(K):
            coords_df[f"topic_{t}"] = W[:, t].round(6)

        coords_path = OUTPUT_DIR / f"umap_coords_K{K}.csv"
        coords_df.to_csv(coords_path, index=False)
        print(f"  Coordinates -> {coords_path.name}")

        # Two-panel figure: (a) original | (b) consensus
        print(f"  Plotting two-panel figure ...")
        make_figure(
            embedding       = embedding,
            bifolio_ids     = bifolio_ids,
            original_order  = original_order,
            consensus_order = consensus_order,
            K               = K,
            out_stem        = OUTPUT_DIR / f"figure3_K{K}",
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
