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
voynich_folio.py
================
Folio-level seriation of the Voynich Manuscript.

Operates directly on doc_id (folio number) with no remapping.
Runs the same NMF ensemble + SA seriation pipeline as sweep.py across all
three distance metrics, with folio 1 forced as the starting point.

Outputs (all in outputs/voynich_folio/):
  results_full.csv              one row per (metric, restart)
  ordering_<metric>.csv         ranked folio sequence per metric
  folio_adjacency_agreement.csv pairwise agreement counts across metrics
  consensus_scaffold.csv        high-confidence adjacency chains (n>=2)

Usage
-----
    python voynich_folio.py

Requires foliation.py and sweep.py in the same directory.
"""

from __future__ import annotations

import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import sweep

# =============================================================================
# CONFIGURATION
# =============================================================================

FOLIATION_CSV       = Path("voynich/vms_rf1b_folio_bifolio.csv")
OUTPUT_DIR          = Path("outputs/voynich_folio")
DISTANCE_METRICS    = ["hellinger", "js", "l1"]
FORCED_FIRST_FOLIO  = 1           # doc_id value to fix at position 0

# NMF — inherited from sweep.py
K_MIN     = sweep.K_MIN
K_MAX     = sweep.K_MAX
NMF_SEEDS = sweep.NMF_SEEDS

# Seriation
N_RESTARTS         = sweep.N_RESTARTS
SA_MOVES           = sweep.SA_MOVES
N_STEPS_MULTIPLIER = 200          # folio problem is larger than bifolio

SOURCE_LABEL = "voynich_folio"

# =============================================================================
# SA WITH FIXED FIRST POSITION
# (identical to voynich_bifolio.py — keeping local so scripts are self-contained)
# =============================================================================

def _sa_seriate_fixed_start(
    D:           np.ndarray,
    init_order:  list[int],
    seed:        int,
    fixed_first: int | None = None,
    n_steps_multiplier: int = 200,
) -> tuple[list[int], float]:
    """
    SA seriation with optional fixed first position.

    fixed_first is the row-index that must remain at position 0.
    All move proposals displacing position 0 are rejected before evaluation.
    """
    local_rng = random.Random(seed)
    n = len(init_order)
    T_init, T_final = sweep._auto_temperature(D, n)

    total_steps   = n * n_steps_multiplier
    cooling_ratio = (T_final / T_init) ** (1.0 / max(total_steps, 1))

    current   = init_order[:]
    cur_cost  = sweep._path_cost(current, D)
    best      = current[:]
    best_cost = cur_cost

    T = T_init
    _EPS = 1e-12

    for _ in range(total_steps):
        move_type = local_rng.choice(SA_MOVES)

        if move_type == "2opt":
            i_min = 1 if fixed_first is not None else 0
            if n - 2 < i_min:
                T *= cooling_ratio
                continue
            i = local_rng.randint(i_min, n - 2)
            j = local_rng.randint(i + 1, n - 1)
            candidate = sweep._apply_2opt(current, i, j)

        else:
            length = int(move_type[2])
            if n < length + 2:
                T *= cooling_ratio
                continue
            i_min = 1 if fixed_first is not None else 0
            if n - length - 1 < i_min:
                T *= cooling_ratio
                continue
            i = local_rng.randint(i_min, n - length - 1)
            j = local_rng.randint(0, n - length - 2)
            if j >= i:
                j += 1
            if j > n - length - 1:
                T *= cooling_ratio
                continue
            candidate = sweep._apply_or_opt(current, i, length, j)
            if fixed_first is not None and candidate[0] != current[0]:
                T *= cooling_ratio
                continue

        cand_cost = sweep._path_cost(candidate, D)
        delta     = cand_cost - cur_cost

        if delta < 0 or local_rng.random() < math.exp(-delta / max(T, _EPS)):
            current  = candidate
            cur_cost = cand_cost
            if cur_cost < best_cost:
                best      = current[:]
                best_cost = cur_cost

        T *= cooling_ratio

    return best, best_cost


# =============================================================================
# LOAD FOLIATION
# =============================================================================

def load_folio_df(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the Voynich foliation CSV. doc_id is used directly as-is.

    Returns
    -------
    df   : token-level DataFrame ready for count matrix construction
    meta : one row per folio with hand/language labels
    """
    raw = pd.read_csv(csv_path, encoding="utf-8-sig")

    missing = {"doc_id", "token"} - set(raw.columns)
    if missing:
        raise ValueError(f"Foliation CSV missing required columns: {missing}")

    raw["doc_id"] = raw["doc_id"].astype(int)

    # Per-folio metadata
    def majority(s):
        m = s.mode()
        return m.iloc[0] if not m.empty else pd.NA

    label_cols = {}
    if "lfd_hand" in raw.columns:
        label_cols["hand"]        = ("lfd_hand",     majority)
    if "rz_lang" in raw.columns:
        label_cols["lang_rz"]     = ("rz_lang",      majority)
    if "currier_lang" in raw.columns:
        label_cols["lang_currier"]= ("currier_lang",  majority)
    if "bifolio_global" in raw.columns:
        label_cols["bifolio_global"] = ("bifolio_global", "first")

    if label_cols:
        meta = (
            raw.groupby("doc_id")
               .agg(**{k: v for k, v in label_cols.items()})
               .reset_index()
        )
    else:
        meta = raw[["doc_id"]].drop_duplicates().reset_index(drop=True)

    return raw[["doc_id", "token"]], meta


# =============================================================================
# SINGLE METRIC RUN
# =============================================================================

def run_metric(
    df:                pd.DataFrame,
    metric:            str,
    forced_first_folio: int | None,
    idx:               int,
    n_total:           int,
) -> tuple[list[dict], list[int], list[int]]:
    """
    Run NMF ensemble + SA seriation for one metric at the folio level.

    Returns
    -------
    results    : list of dicts (one per restart)
    best_order : row-index ordering for the best restart
    folio_ids  : list mapping row index -> doc_id
    """
    t0 = time.time()
    print(f"  [{idx}/{n_total}] folio | metric={metric} ... ", end="", flush=True)

    counts_full = (
        df.groupby(["doc_id", "token"])
          .size()
          .unstack(fill_value=0)
          .sort_index()
    )
    counts = sweep.prune_vocabulary(counts_full)

    if counts.shape[1] == 0:
        print("SKIP (no vocabulary after pruning)")
        return [], [], []

    folio_ids = counts.index.tolist()
    n_docs    = len(folio_ids)

    if n_docs < 2:
        print(f"SKIP (only {n_docs} folio(s))")
        return [], [], []

    # Locate forced-first folio in row index
    forced_row = None
    if forced_first_folio is not None:
        if forced_first_folio in folio_ids:
            forced_row = folio_ids.index(forced_first_folio)
        else:
            print(f"[warn] FORCED_FIRST_FOLIO={forced_first_folio} not found; ignoring.")

    # NMF ensemble
    base_seed = abs(hash(("voynich_folio", metric))) % (2**31)
    doc_topic_by_KS, _, weights_by_KS = sweep.build_nmf_ensemble(
        counts, K_MIN, K_MAX, NMF_SEEDS, base_seed=base_seed
    )

    # Distance matrix
    D = sweep.build_distance_matrix(doc_topic_by_KS, weights_by_KS, metric, n_docs)

    # SVD initial order with forced first
    svd_order = sweep._svd_initial_order(
        doc_topic_by_KS, weights_by_KS, n_docs, K_MAX
    )
    if forced_row is not None and forced_row in svd_order:
        svd_order.remove(forced_row)
        svd_order = [forced_row] + svd_order

    results    = []
    best_order = None
    best_cost  = float("inf")

    for restart in range(N_RESTARTS):
        if restart == 0:
            init = svd_order[:]
        else:
            init = list(sweep.rng.permutation(n_docs))
            if forced_row is not None:
                if forced_row in init:
                    init.remove(forced_row)
                init = [forced_row] + init

        order, cost = _sa_seriate_fixed_start(
            D, init,
            seed=base_seed + restart,
            fixed_first=forced_row,
            n_steps_multiplier=N_STEPS_MULTIPLIER,
        )

        if cost < best_cost:
            best_cost  = cost
            best_order = order[:]

        results.append({
            "source_text":    SOURCE_LABEL,
            "metric":         metric,
            "restart":        restart,
            "n_folios":       n_docs,
            "seriation_cost": round(float(cost), 6),
        })

    elapsed = time.time() - t0
    print(f"done in {elapsed:.1f}s  best_cost={best_cost:.4f}")

    return results, best_order, folio_ids


# =============================================================================
# CONSENSUS HELPERS
# =============================================================================

def build_adjacency_agreement(
    orderings: dict[str, list[int]],
) -> pd.DataFrame:
    """
    For every unordered pair of folios, count how many metric orderings
    place them adjacent (in either direction).

    Returns DataFrame: folio_a, folio_b, n_metrics_adjacent, metrics_agreeing.
    Sorted descending by n_metrics_adjacent.
    """
    pair_counts: dict[tuple, list[str]] = defaultdict(list)

    for metric, order in orderings.items():
        for i in range(len(order) - 1):
            a, b = order[i], order[i + 1]
            pair = (min(a, b), max(a, b))
            pair_counts[pair].append(metric)

    rows = [
        {
            "folio_a":             a,
            "folio_b":             b,
            "n_metrics_adjacent":  len(metrics),
            "metrics_agreeing":    ",".join(sorted(metrics)),
        }
        for (a, b), metrics in pair_counts.items()
    ]

    return (
        pd.DataFrame(rows)
          .sort_values("n_metrics_adjacent", ascending=False)
          .reset_index(drop=True)
    )


def build_consensus_scaffold(
    agreement_df:  pd.DataFrame,
    forced_first:  int | None,
    min_agreement: int = 2,
) -> pd.DataFrame:
    """
    Greedily chain high-confidence adjacencies (n_metrics_adjacent >= min_agreement)
    into a partial sequence starting from forced_first.

    Ambiguities and unplaced folios are flagged explicitly.

    Returns DataFrame: position, folio, confidence, note.
    """
    high = agreement_df[
        agreement_df["n_metrics_adjacent"] >= min_agreement
    ].copy()

    adj: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for _, row in high.iterrows():
        a, b, n = int(row["folio_a"]), int(row["folio_b"]), int(row["n_metrics_adjacent"])
        adj[a].append((b, n))
        adj[b].append((a, n))

    all_folios = set(high["folio_a"]) | set(high["folio_b"])

    placed = set()
    chain  = []

    start = forced_first if (forced_first in all_folios) else (
        next(iter(all_folios)) if all_folios else None
    )

    if start is None:
        return pd.DataFrame(columns=["position", "folio", "confidence", "note"])

    def extend_chain(node: int) -> None:
        placed.add(node)
        chain.append(node)
        for nxt, conf in sorted(adj[node], key=lambda x: -x[1]):
            if nxt not in placed:
                chain.append(("edge", conf))
                extend_chain(nxt)
                return

    extend_chain(start)

    rows = []
    pos  = 1
    i    = 0
    while i < len(chain):
        item = chain[i]
        if isinstance(item, tuple) and item[0] == "edge":
            i += 1
            continue
        conf = None
        if i > 0 and isinstance(chain[i - 1], tuple) and chain[i - 1][0] == "edge":
            conf = chain[i - 1][1]
        n_neighbours = len(adj[item])
        note = ""
        if n_neighbours > 2:
            note = f"ambiguous ({n_neighbours} high-conf neighbours)"
        elif n_neighbours == 2 and pos == 1:
            note = "chain start (2 neighbours)"
        rows.append({
            "position":   pos,
            "folio":      item,
            "confidence": conf if conf is not None else min_agreement,
            "note":       note,
        })
        pos += 1
        i   += 1

    unplaced = all_folios - placed
    if unplaced:
        rows.append({
            "position":   "—",
            "folio":      "GAP",
            "confidence": None,
            "note":       f"unplaced high-conf folios: {sorted(unplaced)}",
        })

    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Voynich Manuscript — folio-level seriation")
    print(f"  Input        : {FOLIATION_CSV}")
    print(f"  Metrics      : {DISTANCE_METRICS}")
    print(f"  Forced first : folio {FORCED_FIRST_FOLIO}")
    print(f"  NMF          : K={K_MIN}–{K_MAX}, {NMF_SEEDS} seeds")
    print(f"  SA           : {N_RESTARTS} restarts, {N_STEPS_MULTIPLIER}× steps")
    print("=" * 60)

    # --- Load ---
    df, meta = load_folio_df(FOLIATION_CSV)

    n_folios = df["doc_id"].nunique()
    n_tokens = len(df)
    print(f"\n  Folios : {n_folios}")
    print(f"  Tokens : {n_tokens}\n")

    # --- Run each metric ---
    all_results: list[dict]      = []
    best_orders: dict[str, list] = {}   # metric -> list of doc_id in order

    n_total = len(DISTANCE_METRICS)
    for idx, metric in enumerate(DISTANCE_METRICS, start=1):
        results, best_order, folio_ids = run_metric(
            df, metric, FORCED_FIRST_FOLIO, idx, n_total
        )
        all_results.extend(results)

        if best_order:
            best_orders[metric] = [folio_ids[i] for i in best_order]

    if not all_results:
        print("No results generated. Check FOLIATION_CSV.")
        return

    # --- Full results CSV ---
    full_df   = pd.DataFrame(all_results)
    full_path = OUTPUT_DIR / "results_full.csv"
    full_df.to_csv(full_path, index=False)
    print(f"\nFull results  -> {full_path}  ({len(full_df)} rows)")

    # --- Per-metric ordering CSVs ---
    for metric, ordered_folios in best_orders.items():
        rows = []
        for rank, fid in enumerate(ordered_folios, start=1):
            row = {"rank": rank, "folio": fid}
            meta_row = meta[meta["doc_id"] == fid]
            if not meta_row.empty:
                row.update(meta_row.iloc[0].to_dict())
            rows.append(row)
        ord_df   = pd.DataFrame(rows)
        ord_path = OUTPUT_DIR / f"ordering_{metric}.csv"
        ord_df.to_csv(ord_path, index=False)
        print(f"Ordering ({metric:>9}) -> {ord_path}")

    # --- Pairwise adjacency agreement ---
    if len(best_orders) > 1:
        agreement_df   = build_adjacency_agreement(best_orders)
        agreement_path = OUTPUT_DIR / "folio_adjacency_agreement.csv"
        agreement_df.to_csv(agreement_path, index=False)
        print(f"Agreement table -> {agreement_path}  ({len(agreement_df)} pairs)")

        # --- Consensus scaffold ---
        scaffold_df   = build_consensus_scaffold(
            agreement_df, forced_first=FORCED_FIRST_FOLIO, min_agreement=2
        )
        scaffold_path = OUTPUT_DIR / "consensus_scaffold.csv"
        scaffold_df.to_csv(scaffold_path, index=False)
        print(f"Consensus       -> {scaffold_path}  ({len(scaffold_df)} entries)")

    # --- Console summary ---
    print("\n--- Best seriation cost by metric ---")
    print(
        full_df.groupby("metric")["seriation_cost"]
               .min()
               .round(4)
               .to_string()
    )

    if len(best_orders) > 1:
        n_agree = (
            agreement_df["n_metrics_adjacent"]
            .value_counts()
            .sort_index(ascending=False)
        )
        print("\n--- Adjacency agreement distribution ---")
        for n, count in n_agree.items():
            print(f"  {n}/3 metrics agree: {count} pairs")

    print("\nDone.")


if __name__ == "__main__":
    main()
