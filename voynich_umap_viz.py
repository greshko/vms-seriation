"""
voynich_umap_viz.py
===================
NMF topic modeling + UMAP dimensionality reduction + SVG seriation visualization
for the Voynich Manuscript at the bifolio level.

For each topic count K in K_VALUES:
  1. Fit NMF on bifolio-level token counts (bifolio_global as doc_id)
  2. Project the K-dimensional topic vectors to 2D via UMAP
  3. Produce two SVG charts per K:
       <K>_original.svg   — seriation line following original ordering
       <K>_consensus.svg  — seriation line following consensus ordering
  4. Write UMAP coordinates to umap_coords_K<K>.csv

All outputs go to OUTPUT_DIR.

Usage
-----
    python voynich_umap_viz.py

Requires: numpy, pandas, scikit-learn, umap-learn, matplotlib
Install:  pip install umap-learn matplotlib
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend; SVG output only
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
import umap

# =============================================================================
# CONFIGURATION
# =============================================================================

FOLIATION_CSV   = Path("voynich/vms_rf1b_folio_bifolio.csv")
CONSENSUS_CSV   = Path("voynich/consensus_vs_original.csv")
OUTPUT_DIR      = Path("outputs/voynich_umap")

K_VALUES        = [8, 12, 16, 20]   # topic counts to visualize

NMF_SEED        = 42
UMAP_SEED       = 42
UMAP_N_NEIGHBORS = 10    # lower = more local structure
UMAP_MIN_DIST    = 0.1

# Visual parameters
FIG_SIZE        = (9, 9)           # inches
DPI             = 150              # only affects rasterized elements; SVG is vector
NODE_SIZE       = 60               # scatter point size
NODE_COLOR      = "#333333"
NODE_ALPHA      = 0.85
LABEL_FONTSIZE  = 7
LINE_ALPHA      = 0.75
LINE_WIDTH      = 1.5
COLORMAP        = "plasma"         # gradient colormap for seriation line
START_MARKER_SIZE = 140            # larger marker for bifolio 1

# =============================================================================
# HELPERS
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
# SVG CHART
# =============================================================================

def make_svg(
    umap_coords:  np.ndarray,        # (n_bifolios, 2)
    bifolio_ids:  list[int],         # parallel to rows of umap_coords
    ordering:     list[int],         # bifolio_global values in sequence order
    title:        str,
    out_path:     Path,
    colormap:     str = COLORMAP,
) -> None:
    """
    Produce one SVG chart:
      - Scatter all bifolios in UMAP space (grey dots, labelled)
      - Draw a gradient line following `ordering` from start to end
      - Highlight bifolio 1 (forced start) with a distinct marker
    """
    bg_to_row = {bg: i for i, bg in enumerate(bifolio_ids)}

    # Only include bifolios that are both in ordering and in umap_coords
    valid_ordering = [bg for bg in ordering if bg in bg_to_row]
    if not valid_ordering:
        print(f"  [warn] No valid bifolios in ordering for {out_path.name}; skipping.")
        return

    xs = umap_coords[:, 0]
    ys = umap_coords[:, 1]

    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    ax.set_aspect("equal", adjustable="datalim")

    # --- Background scatter: all bifolios ---
    ax.scatter(xs, ys,
               s=NODE_SIZE, color=NODE_COLOR, alpha=NODE_ALPHA,
               zorder=3, linewidths=0)

    # --- Labels ---
    for bg, row in bg_to_row.items():
        ax.annotate(
            str(bg),
            (xs[row], ys[row]),
            fontsize=LABEL_FONTSIZE,
            ha="center", va="bottom",
            xytext=(0, 5), textcoords="offset points",
            color="#555555", zorder=4,
        )

    # --- Gradient seriation line ---
    cmap   = cm.get_cmap(colormap)
    n_segs = len(valid_ordering) - 1

    for step in range(n_segs):
        a_bg  = valid_ordering[step]
        b_bg  = valid_ordering[step + 1]
        a_row = bg_to_row[a_bg]
        b_row = bg_to_row[b_bg]
        t     = step / max(n_segs - 1, 1)   # 0.0 (start) -> 1.0 (end)
        color = cmap(t)
        ax.plot(
            [xs[a_row], xs[b_row]],
            [ys[a_row], ys[b_row]],
            color=color, alpha=LINE_ALPHA,
            linewidth=LINE_WIDTH, zorder=2,
            solid_capstyle="round",
        )

    # --- Start marker (bifolio 1) ---
    start_bg = valid_ordering[0]
    if start_bg in bg_to_row:
        sr = bg_to_row[start_bg]
        ax.scatter(xs[sr], ys[sr],
                   s=START_MARKER_SIZE, color=cmap(0.0),
                   marker="*", zorder=5, linewidths=0,
                   label=f"Start (bifolio {start_bg})")

    # --- End marker ---
    end_bg = valid_ordering[-1]
    if end_bg in bg_to_row:
        er = bg_to_row[end_bg]
        ax.scatter(xs[er], ys[er],
                   s=START_MARKER_SIZE * 0.7, color=cmap(1.0),
                   marker="s", zorder=5, linewidths=0,
                   label=f"End (bifolio {end_bg})")

    # --- Colorbar as gradient reference ---
    sm = cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=1, vmax=len(valid_ordering)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Seriation rank", fontsize=9)

    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xlabel("UMAP 1", fontsize=9)
    ax.set_ylabel("UMAP 2", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path.name}")


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

        # SVG: original ordering
        print(f"  Plotting original ordering ...")
        make_svg(
            umap_coords = embedding,
            bifolio_ids = bifolio_ids,
            ordering    = original_order,
            title       = f"Voynich bifolio seriation — original ordering  (K={K})",
            out_path    = OUTPUT_DIR / f"K{K}_original.svg",
        )

        # SVG: consensus ordering
        print(f"  Plotting consensus ordering ...")
        make_svg(
            umap_coords = embedding,
            bifolio_ids = bifolio_ids,
            ordering    = consensus_order,
            title       = f"Voynich bifolio seriation — consensus ordering  (K={K})",
            out_path    = OUTPUT_DIR / f"K{K}_consensus.svg",
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
