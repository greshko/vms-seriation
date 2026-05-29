"""
voynich_bifolio.py
==================
Bifolio-level seriation of the Voynich Manuscript.

Treats each bifolio as a single document by remapping bifolio_global -> doc_id,
then runs the same NMF ensemble + SA seriation pipeline as sweep.py across all
three distance metrics, with bifolio 1 forced as the starting point.

Outputs (all in outputs/voynich_bifolio/):
  results_full.csv               one row per (metric, restart)
  ordering_<metric>.csv          ranked bifolio sequence per metric
  bifolio_adjacency_agreement.csv  pairwise agreement counts across metrics
  consensus_scaffold.csv         high-confidence adjacency chains (n>=2)

Usage
-----
    python voynich_bifolio.py

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

import foliation as fol
import sweep

# =============================================================================
# CONFIGURATION
# =============================================================================

FOLIATION_CSV      = Path("voynich/vms_rf1b_folio_bifolio.csv")
OUTPUT_DIR         = Path("outputs/voynich_bifolio")
DISTANCE_METRICS   = ["hellinger", "js", "l1"]
FORCED_FIRST_BIFOLIO = 1          # bifolio_global value to fix at position 0

# NMF — inherited from sweep.py
K_MIN     = sweep.K_MIN
K_MAX     = sweep.K_MAX
NMF_SEEDS = sweep.NMF_SEEDS

# Seriation
N_RESTARTS           = sweep.N_RESTARTS
SA_MOVES             = sweep.SA_MOVES
N_STEPS_MULTIPLIER   = 400        # more steps given smaller problem size

SOURCE_LABEL = "voynich_bifolio"

# =============================================================================
# SA WITH FIXED FIRST POSITION
# =============================================================================

def _sa_seriate_fixed_start(
    D:          np.ndarray,
    init_order: list[int],
    seed:       int,
    fixed_first: int | None = None,
    n_steps_multiplier: int = 400,
) -> tuple[list[int], float]:
    """
    SA seriation with optional fixed first position.

    If fixed_first is not None, it is the *index into init_order* that must
    remain at position 0. All move proposals that would displace position 0
    are rejected before evaluation.

    Returns best_order, best_cost.
    """
    local_rng = random.Random(seed)
    n = len(init_order)
    T_init, T_final = sweep._auto_temperature(D, n)

    total_steps = n * n_steps_multiplier
    cooling_ratio = (T_final / T_init) ** (1.0 / max(total_steps, 1))

    current   = init_order[:]
    cur_cost  = sweep._path_cost(current, D)
    best      = current[:]
    best_cost = cur_cost

    T = T_init

    for _ in range(total_steps):
        move_type = local_rng.choice(SA_MOVES)

        if move_type == "2opt":
            # If fix_first, i must be >= 1 so position 0 is never inside
            # the reversed segment.
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

            # Or-opt: choose chain start, avoiding position 0 if fixed
            i_min = 1 if fixed_first is not None else 0
            if n - length - 1 < i_min:
                T *= cooling_ratio
                continue
            i = local_rng.randint(i_min, n - length - 1)

            # Insertion position in the shortened sequence
            j = local_rng.randint(0, n - length - 2)
            if j >= i:
                j += 1
            if j > n - length - 1:
                T *= cooling_ratio
                continue

            candidate = sweep._apply_or_opt(current, i, length, j)

            # Reject if the fixed element has moved from position 0
            if fixed_first is not None and candidate[0] != current[0]:
                T *= cooling_ratio
                continue

        cand_cost = sweep._path_cost(candidate, D)
        delta = cand_cost - cur_cost

        _EPS = 1e-12
        if delta < 0 or local_rng.random() < math.exp(-delta / max(T, _EPS)):
            current  = candidate
            cur_cost = cand_cost
            if cur_cost < best_cost:
                best      = current[:]
                best_cost = cur_cost

        T *= cooling_ratio

    return best, best_cost


# =============================================================================
# LOAD AND REMAP FOLIATION
# =============================================================================

def load_bifolio_df(csv_path: Path) -> pd.DataFrame:
    """
    Load the Voynich foliation CSV and remap bifolio_global -> doc_id.

    Each token is now associated with its bifolio rather than its folio,
    so the NMF count matrix pools all tokens from both folios of a bifolio
    into a single document.

    Also returns a metadata DataFrame (one row per bifolio) carrying
    folio membership and majority-vote hand/language labels.
    """
    raw = pd.read_csv(csv_path, encoding="utf-8-sig")

    missing = {"doc_id", "token", "bifolio_global"} - set(raw.columns)
    if missing:
        raise ValueError(f"Foliation CSV missing required columns: {missing}")

    raw["doc_id"]         = raw["doc_id"].astype(int)
    raw["bifolio_global"] = raw["bifolio_global"].astype(int)

    # Remap: bifolio_global becomes the new doc_id
    df = raw.copy()
    df["original_folio"] = df["doc_id"]
    df["doc_id"]         = df["bifolio_global"]
    # bifolio_global now equals doc_id; keep it for adjacency scoring
    # (adjacency of a bifolio with itself is always 1, but the scoring
    # function will treat every "folio" as its own bifolio, which is correct
    # since we want to measure whether the two constituent folios end up
    # adjacent — but at this level each doc IS a bifolio, so we use a
    # trivial bifolio_global = doc_id mapping and don't score adjacency
    # in the same sense as before).
    df["bifolio_global"]  = df["doc_id"]

    return df, raw


def build_bifolio_metadata(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Build a per-bifolio metadata table from the raw foliation.

    Columns: bifolio_global, folio_1, folio_2, n_folios,
             majority_hand, majority_lang_rz, majority_lang_currier
    """
    def majority(s):
        m = s.mode()
        return m.iloc[0] if not m.empty else pd.NA

    # Folios per bifolio
    folio_lists = (
        raw.groupby("bifolio_global")["doc_id"]
           .apply(lambda x: sorted(x.unique()))
           .reset_index()
           .rename(columns={"doc_id": "folios"})
    )
    folio_lists["folio_1"]  = folio_lists["folios"].apply(lambda x: x[0])
    folio_lists["folio_2"]  = folio_lists["folios"].apply(
        lambda x: x[1] if len(x) > 1 else pd.NA
    )
    folio_lists["n_folios"] = folio_lists["folios"].apply(len)

    # Majority-vote labels (only if columns present)
    agg = {"doc_id": "nunique"}
    label_cols = {}
    if "lfd_hand" in raw.columns:
        label_cols["majority_hand"] = ("lfd_hand", majority)
    if "rz_lang" in raw.columns:
        label_cols["majority_lang_rz"] = ("rz_lang", majority)
    if "currier_lang" in raw.columns:
        label_cols["majority_lang_currier"] = ("currier_lang", majority)

    if label_cols:
        label_df = (
            raw.groupby("bifolio_global")
               .agg(**{k: v for k, v in label_cols.items()})
               .reset_index()
        )
        meta = folio_lists.merge(label_df, on="bifolio_global")
    else:
        meta = folio_lists.copy()

    return meta[["bifolio_global", "folio_1", "folio_2", "n_folios"]
                + list(label_cols.keys())]


# =============================================================================
# SINGLE METRIC RUN
# =============================================================================

def run_metric(
    df:          pd.DataFrame,
    metric:      str,
    forced_first_bifolio: int | None,
    idx:         int,
    n_total:     int,
) -> tuple[list[dict], list[int], list[int]]:
    """
    Run NMF ensemble + SA seriation for one metric at the bifolio level.

    Returns
    -------
    results      : list of dicts (one per restart)
    best_order   : row-index ordering for the best restart
    bifolio_ids  : list mapping row index -> bifolio_global
    """
    t0 = time.time()
    print(f"  [{idx}/{n_total}] bifolio | metric={metric} ... ", end="", flush=True)

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

    bifolio_ids = counts.index.tolist()   # list of bifolio_global values
    n_docs      = len(bifolio_ids)

    if n_docs < 2:
        print(f"SKIP (only {n_docs} bifolio(s))")
        return [], [], []

    # Locate the forced-first bifolio in the row index
    forced_row = None
    if forced_first_bifolio is not None:
        if forced_first_bifolio in bifolio_ids:
            forced_row = bifolio_ids.index(forced_first_bifolio)
        else:
            print(f"[warn] forced_first_bifolio={forced_first_bifolio} not found; ignoring.")

    # NMF ensemble
    base_seed = abs(hash(("voynich_bifolio", metric))) % (2**31)
    doc_topic_by_KS, _, weights_by_KS = sweep.build_nmf_ensemble(
        counts, K_MIN, K_MAX, NMF_SEEDS, base_seed=base_seed
    )

    # Distance matrix
    D = sweep.build_distance_matrix(doc_topic_by_KS, weights_by_KS, metric, n_docs)

    # SVD initial order
    svd_order = sweep._svd_initial_order(
        doc_topic_by_KS, weights_by_KS, n_docs, K_MAX
    )

    # Place forced_row at position 0 in svd_order
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
            # Ensure forced_row is at position 0
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
            "n_bifolios":     n_docs,
            "seriation_cost": round(float(cost), 6),
        })

    elapsed = time.time() - t0
    print(f"done in {elapsed:.1f}s  best_cost={best_cost:.4f}")

    return results, best_order, bifolio_ids


# =============================================================================
# CONSENSUS HELPERS
# =============================================================================

def build_adjacency_agreement(
    orderings: dict[str, list[int]],
) -> pd.DataFrame:
    """
    For every unordered pair of bifolios, count how many metric orderings
    place them adjacent (in either direction).

    Parameters
    ----------
    orderings : dict metric -> list of bifolio_global values in recovered order

    Returns
    -------
    DataFrame with columns:
        bifolio_a, bifolio_b, n_metrics_adjacent, metrics_agreeing
    Sorted descending by n_metrics_adjacent.
    """
    pair_counts: dict[tuple, list[str]] = defaultdict(list)

    for metric, order in orderings.items():
        for i in range(len(order) - 1):
            a, b = order[i], order[i + 1]
            pair = (min(a, b), max(a, b))
            pair_counts[pair].append(metric)

    rows = []
    for (a, b), metrics in pair_counts.items():
        rows.append({
            "bifolio_a":           a,
            "bifolio_b":           b,
            "n_metrics_adjacent":  len(metrics),
            "metrics_agreeing":    ",".join(sorted(metrics)),
        })

    df = pd.DataFrame(rows).sort_values(
        "n_metrics_adjacent", ascending=False
    ).reset_index(drop=True)

    return df


def build_consensus_scaffold(
    agreement_df: pd.DataFrame,
    forced_first: int | None,
    min_agreement: int = 2,
) -> pd.DataFrame:
    """
    Greedily chain high-confidence adjacencies (n_metrics_adjacent >= min_agreement)
    into a partial sequence, starting from forced_first if provided.

    Gaps (bifolios not yet placed) are marked explicitly.
    Ambiguities (a bifolio appearing in multiple high-confidence pairs) are
    flagged rather than resolved automatically.

    Returns a DataFrame with columns:
        position, bifolio_global, confidence, note
    where confidence is the n_metrics_adjacent of the edge leading to this node,
    and note flags ambiguities or gaps.
    """
    high = agreement_df[
        agreement_df["n_metrics_adjacent"] >= min_agreement
    ].copy()

    # Build adjacency sets from high-confidence pairs
    adj: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for _, row in high.iterrows():
        a, b, n = int(row["bifolio_a"]), int(row["bifolio_b"]), int(row["n_metrics_adjacent"])
        adj[a].append((b, n))
        adj[b].append((a, n))

    # All bifolios that appear in any high-confidence pair
    all_bifolios = set(high["bifolio_a"]) | set(high["bifolio_b"])

    # Greedy chain construction
    placed = set()
    chain  = []

    # Start from forced_first if it's in the high-confidence graph
    start = forced_first if (forced_first in all_bifolios) else (
        next(iter(all_bifolios)) if all_bifolios else None
    )

    if start is None:
        return pd.DataFrame(columns=["position", "bifolio_global", "confidence", "note"])

    def extend_chain(node):
        """Follow the highest-confidence unvisited neighbour."""
        placed.add(node)
        chain.append(node)
        # Sort neighbours by descending confidence
        neighbours = sorted(adj[node], key=lambda x: -x[1])
        for nxt, conf in neighbours:
            if nxt not in placed:
                chain.append(("edge", conf))
                extend_chain(nxt)
                return

    extend_chain(start)

    # Build output rows from chain
    rows = []
    pos  = 1
    i    = 0
    while i < len(chain):
        item = chain[i]
        if isinstance(item, tuple) and item[0] == "edge":
            i += 1
            continue
        # Look back for the edge confidence
        conf = None
        if i > 0 and isinstance(chain[i - 1], tuple) and chain[i - 1][0] == "edge":
            conf = chain[i - 1][1]

        # Flag if this bifolio has multiple high-confidence neighbours
        n_neighbours = len(adj[item])
        note = ""
        if n_neighbours > 2:
            note = f"ambiguous ({n_neighbours} high-conf neighbours)"
        elif n_neighbours == 2 and pos == 1:
            note = "chain start (2 neighbours)"

        rows.append({
            "position":       pos,
            "bifolio_global": item,
            "confidence":     conf if conf is not None else min_agreement,
            "note":           note,
        })
        pos += 1
        i   += 1

    # Append unplaced bifolios as gaps
    unplaced = all_bifolios - placed
    if unplaced:
        rows.append({
            "position":       "—",
            "bifolio_global": "GAP",
            "confidence":     None,
            "note":           f"unplaced high-conf bifolios: {sorted(unplaced)}",
        })

    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Voynich Manuscript — bifolio-level seriation")
    print(f"  Input        : {FOLIATION_CSV}")
    print(f"  Metrics      : {DISTANCE_METRICS}")
    print(f"  Forced first : bifolio {FORCED_FIRST_BIFOLIO}")
    print(f"  NMF          : K={K_MIN}–{K_MAX}, {NMF_SEEDS} seeds")
    print(f"  SA           : {N_RESTARTS} restarts, {N_STEPS_MULTIPLIER}× steps")
    print("=" * 60)

    # --- Load and remap ---
    df, raw = load_bifolio_df(FOLIATION_CSV)
    meta    = build_bifolio_metadata(raw)

    n_bifolios = df["doc_id"].nunique()
    n_tokens   = len(df)
    print(f"\n  Bifolios : {n_bifolios}")
    print(f"  Tokens   : {n_tokens}")
    print(f"  Unpaired : {(meta['n_folios'] == 1).sum()} bifolio(s) with one folio\n")

    # --- Run each metric ---
    all_results:  list[dict]       = []
    best_orders:  dict[str, list]  = {}   # metric -> list of bifolio_global in order
    bifolio_ids_per_metric: dict   = {}

    n_total = len(DISTANCE_METRICS)
    for idx, metric in enumerate(DISTANCE_METRICS, start=1):
        results, best_order, bifolio_ids = run_metric(
            df, metric, FORCED_FIRST_BIFOLIO, idx, n_total
        )
        all_results.extend(results)

        if best_order:
            # Convert row indices back to bifolio_global values
            ordered_bifolios = [bifolio_ids[i] for i in best_order]
            best_orders[metric]              = ordered_bifolios
            bifolio_ids_per_metric[metric]   = bifolio_ids

    if not all_results:
        print("No results generated. Check FOLIATION_CSV.")
        return

    # --- Full results CSV ---
    full_df   = pd.DataFrame(all_results)
    full_path = OUTPUT_DIR / "results_full.csv"
    full_df.to_csv(full_path, index=False)
    print(f"\nFull results  -> {full_path}  ({len(full_df)} rows)")

    # --- Per-metric ordering CSVs ---
    for metric, ordered_bifolios in best_orders.items():
        rows = []
        for rank, bg in enumerate(ordered_bifolios, start=1):
            row = {"rank": rank, "bifolio_global": bg}
            meta_row = meta[meta["bifolio_global"] == bg]
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
        agreement_path = OUTPUT_DIR / "bifolio_adjacency_agreement.csv"
        agreement_df.to_csv(agreement_path, index=False)
        print(f"Agreement table -> {agreement_path}  ({len(agreement_df)} pairs)")

        # --- Consensus scaffold ---
        scaffold_df   = build_consensus_scaffold(
            agreement_df, forced_first=FORCED_FIRST_BIFOLIO, min_agreement=2
        )
        scaffold_path = OUTPUT_DIR / "consensus_scaffold.csv"
        scaffold_df.to_csv(scaffold_path, index=False)
        print(f"Consensus       -> {scaffold_path}  ({len(scaffold_df)} entries)")

    # --- Console summary ---
    print("\n--- Best seriation cost by metric ---")
    best_costs = (
        full_df.groupby("metric")["seriation_cost"]
               .min()
               .round(4)
    )
    print(best_costs.to_string())

    # --- Agreement summary ---
    if len(best_orders) > 1:
        n_agree = agreement_df["n_metrics_adjacent"].value_counts().sort_index(ascending=False)
        print("\n--- Adjacency agreement distribution ---")
        for n, count in n_agree.items():
            print(f"  {n}/3 metrics agree: {count} pairs")

    print("\nDone.")


if __name__ == "__main__":
    main()
