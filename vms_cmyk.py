# Copyright (c) 2026, Michael A. Greshko
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software, datasets, and associated documentation files (the "Software
# and Datasets"), to deal in the Software and Datasets without restriction,
# including without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software and Datasets, and to
# permit persons to whom the Software is furnished to do so, subject to the
# following conditions:
# 
# - The above copyright notice and this permission notice shall be included
#   in all copies or substantial portions of the Software and Datasets.
# - Any publications making use of the Software and Datasets, or any substantial
#   portions thereof, shall cite the Software and Datasets's original publication:
# 
# > Greshko, Michael A. (2026). Algorithmic seriation suggests the Voynich
#   Manuscript is most consistent with an all-singulion collation, informing
#   a bifolium-level reordering of the codex. Accepted to the Second International
#   Conference on the Voynich Manuscript 2026.
#   
# THE SOFTWARE AND DATASETS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
# EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE AND DATASETS.

"""
vms_cmyk.py

Builds a multi-row CMYK topic-progression comparison for the VMS token CSV:

  Row 1 - by folio (doc_id), left-to-right in current folio order
  Row 2 - by bifolio (bifolio_global), left-to-right in current physical order
  Rows 3..N-1 - the SAME bifolio colors as row 2, re-ordered left-to-right by
          each candidate seriation column in a rank/original/<seriations...>
          CSV (e.g. hellinger, js, l1 distance-based candidate orderings)
  Row N - re-ordered by the "consensus" reconstructed order (same CSV)

All bifolio-level rows (2..N) share ONE NMF fit over the 52 bifolio-level
chunks — every row after row 2 is a pure permutation of row 2's colors, not
a re-fit, since comparing seriations is a question of *ordering* the same
physical units, not different content.

Usage:
    python vms_seriation_rows.py vms.csv seriations.csv --out out.png \\
        --seriation-cols hellinger js l1 --consensus-col consensus
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, "/mnt/user-data/outputs")
from topic_cmyk import chunk_by_doc_id, fit_topics, topic_mixture_to_cmyk, cmyk_to_rgb, N_TOPICS


def slugify(title: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", title.lower())).strip("_")


def save_stripe(colors, out_dir: str, name: str, width_in: float = 12.0, dpi: int = 300):
    """Save a single label-free 'climate stripes' style color strip, at a
    fixed 3:2 aspect ratio, as both a high-resolution PNG and an SVG. No
    axes, ticks, titles, or margins — just the colors, edge to edge, so the
    result drops cleanly into a slide."""
    os.makedirs(out_dir, exist_ok=True)
    height_in = width_in * 2 / 3  # enforce 3:2 (width:height)

    fig = plt.figure(figsize=(width_in, height_in), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])  # axes fill the entire figure, no margins
    n = len(colors)
    for i, color in enumerate(colors):
        ax.add_patch(Rectangle((i, 0), 1, 1, color=color))
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.axis("off")

    png_path = os.path.join(out_dir, f"{name}.png")
    svg_path = os.path.join(out_dir, f"{name}.svg")
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(svg_path)
    plt.close(fig)
    return png_path, svg_path


def compute_row(csv_path, group_col, min_df=2, max_df=0.8):
    chunks, labels, group_ids = chunk_by_doc_id(csv_path, group_col=group_col)
    doc_topic, nmf, vectorizer = fit_topics(
        chunks, n_topics=N_TOPICS, stopwords="none",
        min_df=min_df, max_df=max_df,
        token_pattern=r"(?u)\S+", lowercase=False,
    )
    cmyk = topic_mixture_to_cmyk(doc_topic)
    rgb = cmyk_to_rgb(cmyk)
    return rgb, labels, group_ids


def reorder_by_column(order_df, col, gid_to_color, gid_to_label):
    ordered = order_df.sort_values("rank")[col]
    missing = set(ordered) - set(gid_to_color)
    if missing:
        raise ValueError(f"Column '{col}' references bifolio ids not in the VMS data: {sorted(missing)}")
    colors = np.array([gid_to_color[v] for v in ordered])
    labels = [gid_to_label[v] for v in ordered]
    return colors, labels


def plot_rows(rows, out_path):
    """rows: list of (title, rgb_colors, labels) triples, top to bottom."""
    n_max = max(len(r[1]) for r in rows)
    fig_width = min(max(10, n_max * 0.35), 42)
    dpi = 150 if n_max <= 70 else max(80, int(150 * 70 / n_max))

    fig, axes = plt.subplots(len(rows), 1, figsize=(fig_width, 2.0 * len(rows)))
    if len(rows) == 1:
        axes = [axes]

    for ax, (title, colors, labels) in zip(axes, rows):
        n = len(colors)
        for i, color in enumerate(colors):
            ax.add_patch(Rectangle((i, 0), 1, 1, color=color))
        ax.set_xlim(0, n)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xticks(np.arange(n) + 0.5)
        ax.set_xticklabels(labels, rotation=90, ha="center", fontsize=6)
        ax.set_title(title, fontsize=11, loc="left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    print(f"Saved visualization to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vms_csv", help="Path to the VMS token CSV (token, doc_id, bifolio_global, ...)")
    parser.add_argument("order_csv", help="Path to the rank/original/<seriation columns...> CSV")
    parser.add_argument("--out", default="./outputs/vms_cmyk.png")
    parser.add_argument("--seriation-cols", nargs="*", default=["hellinger", "js", "l1"],
                         help="Candidate seriation columns to slot between the physical order and consensus")
    parser.add_argument("--consensus-col", default="consensus")
    parser.add_argument("--stripes-dir", default=None,
                         help="Folder for label-free 3:2 PNG/SVG 'climate stripes' per row "
                              "(default: a 'stripes' subfolder next to --out)")
    parser.add_argument("--stripes-width-in", type=float, default=12.0,
                         help="Width in inches of each stripe image (height is fixed at 2/3 of this, for 3:2)")
    parser.add_argument("--stripes-dpi", type=int, default=300)
    args = parser.parse_args()

    print("Fitting topics on folio-level chunks...")
    folio_rgb, folio_labels, _ = compute_row(args.vms_csv, group_col="doc_id")

    print("Fitting topics on bifolio-level chunks...")
    bifolio_rgb, bifolio_labels, bifolio_gids = compute_row(args.vms_csv, group_col="bifolio_global")

    gid_to_color = {gid: color for gid, color in zip(bifolio_gids, bifolio_rgb)}
    gid_to_label = {gid: label for gid, label in zip(bifolio_gids, bifolio_labels)}

    order_df = pd.read_csv(args.order_csv)

    rows = [
        ("By folio (current order)", folio_rgb, folio_labels),
        ("By bifolio (current physical order)", bifolio_rgb, bifolio_labels),
    ]
    for col in args.seriation_cols:
        colors, labels = reorder_by_column(order_df, col, gid_to_color, gid_to_label)
        rows.append((f"By bifolio, candidate seriation: {col}", colors, labels))
    colors, labels = reorder_by_column(order_df, args.consensus_col, gid_to_color, gid_to_label)
    rows.append(("By bifolio, consensus reconstructed order", colors, labels))

    plot_rows(rows, args.out)

    stripes_dir = args.stripes_dir or os.path.join(os.path.dirname(args.out) or ".", "stripes")
    print(f"\nWriting label-free 3:2 stripe images to {stripes_dir}/")
    for i, (title, colors, _labels) in enumerate(rows, start=1):
        name = f"{i:02d}_{slugify(title)}"
        png_path, svg_path = save_stripe(
            colors, stripes_dir, name, width_in=args.stripes_width_in, dpi=args.stripes_dpi
        )
        print(f"  {title} -> {os.path.basename(png_path)}, {os.path.basename(svg_path)}")


if __name__ == "__main__":
    main()

