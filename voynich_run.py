"""
voynich_run.py
==============
Orchestrator for running the seriation sweep on the Voynich Manuscript
using a predetermined foliation CSV.

Unlike sweep.py, which generates synthetic foliations from plain text,
this script:
  1. Loads a fixed foliation CSV via foliation.load_predetermined_foliation()
  2. Runs the same NMF ensemble + SA seriation pipeline as sweep.py
  3. Sweeps only over distance metrics (not words_per_folio or
     bifolios_per_quire, since those are fixed by the manuscript itself)
  4. Writes results to outputs/voynich/

The foliation CSV must contain at minimum:
    doc_id         int   folio number
    token          str   transliteration token (not re-cleaned)
    bifolio_global int   unique bifolio identifier across the manuscript

Usage
-----
    python voynich_run.py

Requires foliation.py and sweep.py in the same directory.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

import foliation as fol
import sweep

# =============================================================================
# CONFIGURATION
# =============================================================================

FOLIATION_CSV = Path("voynich/vms_rf1b_folio_bifolio.csv")
OUTPUT_DIR    = Path("outputs/voynich")

# Sweep only over metrics; foliation is fixed
DISTANCE_METRICS = ["hellinger", "js", "l1"]

# NMF — same settings as sweep.py
K_MIN     = sweep.K_MIN
K_MAX     = sweep.K_MAX
NMF_SEEDS = sweep.NMF_SEEDS

# Seriation — same settings as sweep.py
N_RESTARTS = sweep.N_RESTARTS

# Source label used in output files and result rows
SOURCE_LABEL = "voynich_rf1b"

# =============================================================================
# MAIN
# =============================================================================

def run_voynich_metric(
    df:        pd.DataFrame,
    metric:    str,
    folio_meta: dict,
    baseline:  float,
    summary:   dict,
    idx:       int,
    n_total:   int,
) -> list[dict]:
    """
    Run NMF ensemble + seriation for one distance metric on the Voynich data.
    Returns a list of result dicts, one per restart.
    """
    t0 = time.time()
    print(f"  [{idx}/{n_total}] voynich | metric={metric} ... ", end="", flush=True)

    # --- Count matrix ---
    counts_full = (
        df.groupby(["doc_id", "token"])
          .size()
          .unstack(fill_value=0)
          .sort_index()
    )
    counts = sweep.prune_vocabulary(counts_full)

    if counts.shape[1] == 0:
        print("SKIP (no vocabulary after pruning)")
        return []

    doc_ids = counts.index.tolist()
    n_docs  = len(doc_ids)

    if n_docs < 2:
        print(f"SKIP (only {n_docs} folio(s))")
        return []

    # --- NMF ensemble ---
    base_seed = abs(hash(("voynich", metric))) % (2**31)
    doc_topic_by_KS, _, weights_by_KS = sweep.build_nmf_ensemble(
        counts, K_MIN, K_MAX, NMF_SEEDS, base_seed=base_seed
    )

    # --- Distance matrix ---
    D = sweep.build_distance_matrix(doc_topic_by_KS, weights_by_KS, metric, n_docs)

    # --- Seriation: multiple restarts ---
    svd_order = sweep._svd_initial_order(
        doc_topic_by_KS, weights_by_KS, n_docs, K_MAX
    )

    results = []
    for restart in range(N_RESTARTS):
        init = svd_order[:] if restart == 0 else list(sweep.rng.permutation(n_docs))

        order, cost = sweep._sa_seriate(
            D, init,
            seed=base_seed + restart,
            n_steps_multiplier=200,
        )

        adj_rate = sweep.bifolio_adjacency_rate(order, doc_ids, folio_meta)

        results.append({
            "source_text":        SOURCE_LABEL,
            "metric":             metric,
            "restart":            restart,
            "n_folios":           n_docs,
            "n_bifolios":         summary["n_bifolios"],
            "n_quires":           summary["n_quires"],
            "bifolio_adj_rate":   round(adj_rate, 6),
            "random_baseline":    round(baseline, 6),
            "lift_over_baseline": round(adj_rate - baseline, 6),
            "seriation_cost":     round(float(cost), 6),
        })

    elapsed = time.time() - t0
    best_rate = max(r["bifolio_adj_rate"] for r in results)
    print(f"done in {elapsed:.1f}s  best_adj={best_rate:.4f}  baseline={baseline:.4f}")

    return results


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Voynich Manuscript — seriation run")
    print(f"  Input  : {FOLIATION_CSV}")
    print(f"  Metrics: {DISTANCE_METRICS}")
    print(f"  NMF    : K={K_MIN}–{K_MAX}, {NMF_SEEDS} seeds")
    print(f"  SA     : {N_RESTARTS} restarts")
    print("=" * 60)

    # --- Load predetermined foliation ---
    # utf-8-sig strips the BOM that Excel-style CSVs often carry,
    # which would otherwise corrupt the first column name to '\ufefftoken'.
    raw = pd.read_csv(FOLIATION_CSV, encoding="utf-8-sig")

    missing = {"doc_id", "token", "bifolio_global"} - set(raw.columns)
    if missing:
        raise ValueError(f"Foliation CSV missing required columns: {missing}")

    optional = ["bifolio_id", "quire", "folio_in_quire", "side", "mg_section"]
    for col in optional:
        if col not in raw.columns:
            raw[col] = pd.NA

    raw["doc_id"]         = raw["doc_id"].astype(int)
    raw["bifolio_global"] = raw["bifolio_global"].astype(int)

    df = raw[["doc_id", "token", "bifolio_id", "quire",
              "folio_in_quire", "side", "bifolio_global", "mg_section"]]
    summary = fol.foliation_summary(df)
    baseline = fol.random_baseline(df)

    print(f"\n  Folios   : {summary['n_folios']}")
    print(f"  Bifolios : {summary['n_bifolios']}")
    print(f"  Quires   : {summary['n_quires']}")
    print(f"  Tokens   : {summary['n_tokens']}")
    print(f"  Random baseline adjacency rate: {baseline:.6f}\n")

    folio_meta = (
        df[["doc_id", "bifolio_global"]]
        .drop_duplicates("doc_id")
        .set_index("doc_id")["bifolio_global"]
        .to_dict()
    )

    # --- Sweep over metrics ---
    all_results: list[dict] = []
    n_total = len(DISTANCE_METRICS)

    for idx, metric in enumerate(DISTANCE_METRICS, start=1):
        results = run_voynich_metric(
            df, metric, folio_meta, baseline, summary, idx, n_total
        )
        all_results.extend(results)

    if not all_results:
        print("No results generated. Check FOLIATION_CSV.")
        return

    # --- Full results CSV ---
    full_df = pd.DataFrame(all_results)
    full_path = OUTPUT_DIR / "results_full.csv"
    full_df.to_csv(full_path, index=False)
    print(f"\nFull results  -> {full_path}  ({len(full_df)} rows)")

    # --- Summary CSV ---
    summary_df = (
        full_df
        .groupby("metric")
        .agg(
            bifolio_adj_rate_mean   = ("bifolio_adj_rate",   "mean"),
            bifolio_adj_rate_sd     = ("bifolio_adj_rate",   "std"),
            bifolio_adj_rate_max    = ("bifolio_adj_rate",   "max"),
            random_baseline         = ("random_baseline",    "first"),
            lift_over_baseline_mean = ("lift_over_baseline", "mean"),
            seriation_cost_mean     = ("seriation_cost",     "mean"),
            n_folios                = ("n_folios",           "first"),
            n_bifolios              = ("n_bifolios",         "first"),
            n_restarts              = ("restart",            "count"),
        )
        .reset_index()
        .round(6)
    )
    summary_path = OUTPUT_DIR / "results_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary       -> {summary_path}")

    # --- Console summary ---
    print("\n--- Voynich adjacency rate by metric ---")
    print(summary_df[["metric", "bifolio_adj_rate_mean", "bifolio_adj_rate_sd",
                       "bifolio_adj_rate_max", "random_baseline",
                       "lift_over_baseline_mean"]].to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
