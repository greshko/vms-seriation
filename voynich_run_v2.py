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
voynich_run_v2.py
=================
Run the sweep_v2 pipeline on the Voynich Manuscript (VMS) using its
predetermined foliation, and write results in the schema of sweep_v2's
results_full.csv so that violin_v2.py can use them as the VMS comparison point.

Pipeline (identical to sweep_v2, via its own functions): foliation CSV ->
vocabulary pruning -> NMF ensemble (TF-IDF, K = K_MIN..K_MAX, weighted by
inverse reconstruction error) -> weighted pairwise distance matrix ->
SA seriation (restart 0 from the SVD order, others from random permutations)
-> bifolio adjacency rate -> CAROB = adjacency rate / random baseline.

Differences from voynich_run.py
-------------------------------
* CAROB is computed, using the CORRECTED random baseline 2P / (n (n-1))
  (sweep_v2.exact_random_baseline); the old fol.random_baseline value is kept
  as random_baseline_foliation for comparison.
* Output columns follow sweep_v2's results_full.csv (see below), including
  `best_cost_restart`, so violin_v2's --best-cost-only and --metrics filters
  work on the VMS rows exactly as they do on the baseline texts.
* The NMF ensemble is fitted ONCE and shared across metrics (it does not depend
  on the metric), and the initial orders are shared across metrics, as in
  sweep_v2. Pairwise distances are vectorised (verified against sweep.py).
* Seeding uses numpy SeedSequence instead of hash(), so results are
  reproducible across processes.
* K is capped at min(K_MAX, n_folios, n_vocabulary), as in sweep_v2.
* Tokens are read verbatim: values that pandas would treat as missing
  ("nan", "NA", "null", ...) are kept as words rather than silently dropped.
* --match-sweep copies restarts / NMF seeds / SA steps / K range / metrics from a
  sweep_v2 run's config.json, so VMS and baseline are processed identically,
  and reports how the VMS token count compares with the sweep's window length.

Output schema (OUTPUT_DIR/results_full.csv; one row per metric x restart)
-------------
  source_text, window_idx (0), window_start (0), bifolios_per_quire (0),
  metric, restart, init, n_folios, n_quires, n_bifolios, k_min_used,
  k_max_used, bifolio_adj_rate, random_baseline, random_baseline_foliation,
  carob, lift_over_baseline, seriation_cost, best_cost_restart

  * bifolios_per_quire = 0 is a sentinel meaning "not applicable: the
    manuscript's real foliation is used". violin_v2 requires an integer here,
    and a constant keeps all VMS rows in one group (it collapses them to a single
    value, N = 1).
  * words_per_folio is deliberately ABSENT: folio lengths are fixed by the
    manuscript. (violin_v2's --wpf filter therefore leaves these rows alone;
    its --metrics and --best-cost-only filters do apply to them.)
  * window_idx = 0 / window_start = 0: the whole manuscript is the single "window".

Other outputs: results_summary.csv (per metric, same columns as sweep_v2's
summaries) and config.json (settings, input file hash, foliation structure).

Usage
-----
    python voynich_run_v2.py
    python voynich_run_v2.py --foliation voynich/vms_rf1b_folio_bifolio.csv \\
        --match-sweep outputs_v2/mytext          # copy settings from a sweep_v2 run
    python violin_v2.py --input outputs_v2/results_full_all.csv \\
        --vms-input outputs_v2/voynich/results_full.csv

The foliation CSV needs: doc_id (int folio number), token (str, not re-cleaned),
bifolio_global (int). Optional: bifolio_id, quire, folio_in_quire, side, mg_section.

Requires foliation.py, sweep.py and sweep_v2.py in the same directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import foliation as fol
import sweep as sw
import sweep_v2 as sv2

# =============================================================================
# CONFIGURATION  (defaults; most can be overridden on the command line)
# =============================================================================

FOLIATION_CSV = Path("voynich/vms_rf1b_folio_bifolio.csv")
OUTPUT_DIR    = Path("outputs_v2") / "voynich"
SOURCE_LABEL  = "voynich_rf1b"      # should match violin_v2's --vms-pattern

# Defaults come from sweep_v2 so that VMS and baseline texts are processed alike
DISTANCE_METRICS = list(sv2.DISTANCE_METRICS)
K_MIN, K_MAX     = sv2.K_MIN, sv2.K_MAX
NMF_SEEDS        = sv2.NMF_SEEDS
N_RESTARTS       = sv2.N_RESTARTS
SA_STEP_MULT     = sv2.SA_STEP_MULT
SEED             = sv2.GLOBAL_SEED

BPQ_NOT_APPLICABLE = 0       # sentinel for bifolios_per_quire (predetermined foliation)
_VMS_PATTERN = re.compile(r"voynich|vms", re.IGNORECASE)   # violin_v2's default --vms-pattern
_SCRIPT_VERSION = 2


# =============================================================================
# LOADING
# =============================================================================

def load_foliation(csv_path: Path) -> pd.DataFrame:
    """Load the predetermined foliation CSV (see module docstring for columns)."""
    # utf-8-sig strips the BOM that Excel-style CSVs often carry, which would
    # otherwise corrupt the first column name. keep_default_na=False reads tokens
    # verbatim (e.g. "nan" / "NA" stay words); only empty cells are missing.
    raw = pd.read_csv(csv_path, encoding="utf-8-sig", keep_default_na=False,
                      na_values=[""], dtype={"token": str})
    missing = {"doc_id", "token", "bifolio_global"} - set(raw.columns)
    if missing:
        raise ValueError(f"Foliation CSV missing required columns: {missing}")

    n_empty = int(raw["token"].isna().sum())
    if n_empty:
        print(f"  WARNING: dropping {n_empty} row(s) with an empty token.")
        raw = raw.dropna(subset=["token"])

    for col in ["bifolio_id", "quire", "folio_in_quire", "side", "mg_section"]:
        if col not in raw.columns:
            raw[col] = pd.NA

    raw["doc_id"] = raw["doc_id"].astype(int)
    raw["bifolio_global"] = raw["bifolio_global"].astype(int)

    per_folio = raw.groupby("doc_id")["bifolio_global"].nunique()
    if (per_folio > 1).any():
        bad = per_folio[per_folio > 1].index.tolist()[:10]
        raise ValueError(f"Some folios map to more than one bifolio_global: {bad}")

    return raw[["doc_id", "token", "bifolio_id", "quire", "folio_in_quire",
                "side", "bifolio_global", "mg_section"]]


def load_sweep_settings(path: Path) -> dict:
    """Read settings from a sweep_v2 run: a text's output folder or its config.json."""
    p = path / "config.json" if path.is_dir() else path
    if not p.exists():
        raise SystemExit(f"--match-sweep: {p} not found. Point to a text's folder "
                         f"inside the sweep_v2 output (e.g. outputs_v2/<text>) or its config.json.")
    cfg = json.loads(p.read_text())
    need = {"restarts", "nmf_seeds", "sa_mult", "k", "metrics"}
    if need - set(cfg):
        raise SystemExit(f"--match-sweep: {p} lacks keys {sorted(need - set(cfg))}; "
                         f"is it a sweep_v2 config.json?")
    return cfg


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# =============================================================================
# RUN
# =============================================================================

def run_vms(df: pd.DataFrame, label: str, metrics: list[str], k_min_cfg: int,
            k_max_cfg: int, nmf_seeds: int, restarts: int, sa_mult: int,
            seed: int) -> tuple[list[dict], dict]:
    """Full pipeline on the VMS. Returns (rows, info) where info holds
    foliation-structure facts for config.json."""
    summ = fol.foliation_summary(df)
    folio_bifolio = (df[["doc_id", "bifolio_global"]].drop_duplicates("doc_id")
                     .set_index("doc_id")["bifolio_global"].to_dict())
    base_exact = sv2.exact_random_baseline(folio_bifolio)
    base_fol = fol.random_baseline(df)

    quires = df.drop_duplicates("doc_id")["quire"]
    n_quires = int(quires.nunique()) if quires.notna().any() else float("nan")
    sizes = Counter(Counter(folio_bifolio.values()).values())
    per_folio = df.groupby("doc_id").size()

    info = dict(
        n_tokens=int(summ["n_tokens"]), n_folios=int(summ["n_folios"]),
        n_bifolios=int(summ["n_bifolios"]),
        n_quires=None if n_quires != n_quires else n_quires,
        bifolio_size_counts={int(k): int(v) for k, v in sorted(sizes.items())},
        tokens_per_folio=dict(min=int(per_folio.min()), median=float(per_folio.median()),
                              max=int(per_folio.max())),
        random_baseline_exact=base_exact, random_baseline_foliation=base_fol,
    )

    print(f"\n  Tokens   : {info['n_tokens']}")
    print(f"  Folios   : {info['n_folios']}   (tokens/folio min/median/max = "
          f"{info['tokens_per_folio']['min']}/{info['tokens_per_folio']['median']:.0f}/"
          f"{info['tokens_per_folio']['max']})")
    print(f"  Bifolios : {info['n_bifolios']}   (size -> count: {info['bifolio_size_counts']})")
    print(f"  Quires   : {n_quires}")
    print(f"  Random baseline adjacency rate: {base_exact:.6f} exact "
          f"({base_fol:.6f} by foliation.random_baseline)\n")

    # --- Counts + pruning ---
    counts = sw.prune_vocabulary(
        df.groupby(["doc_id", "token"]).size().unstack(fill_value=0).sort_index())
    doc_ids = counts.index.tolist()
    n_docs = len(doc_ids)
    if counts.shape[1] == 0 or n_docs < 3:
        raise SystemExit("Cannot run: no vocabulary after pruning, or fewer than 3 folios.")

    # --- Seeds: deterministic, independent of hash() and of process ---
    st = np.random.SeedSequence([seed, zlib.crc32(label.encode())]).generate_state(4)
    nmf_seed = int(st[0] % 2**31)
    init_rng = np.random.default_rng(int(st[1]))
    sw.rng = np.random.default_rng(int(st[2]))        # used by sw._auto_temperature
    sa_seed = int(st[3] % 2**31)

    # --- NMF ensemble (once; independent of metric) ---
    k_max = min(k_max_cfg, n_docs, counts.shape[1])
    k_min = min(k_min_cfg, k_max)
    t0 = time.time()
    print(f"  NMF ensemble K={k_min}-{k_max} x{nmf_seeds} seeds ... ", end="", flush=True)
    dt, _errs, wts = sw.build_nmf_ensemble(counts, k_min, k_max, nmf_seeds,
                                           base_seed=nmf_seed)
    print(f"done in {time.time() - t0:.1f}s")
    svd_order = sw._svd_initial_order(dt, wts, n_docs, k_max)
    inits = [svd_order[:]] + [[int(x) for x in init_rng.permutation(n_docs)]
                              for _ in range(restarts - 1)]

    # --- Per metric: distances + SA restarts ---
    rows: list[dict] = []
    for i, metric in enumerate(metrics, 1):
        t0 = time.time()
        print(f"  [{i}/{len(metrics)}] metric={metric} ... ", end="", flush=True)
        D = sv2.weighted_distance_matrix(dt, wts, metric, n_docs)
        results = []
        for r, init in enumerate(inits):
            order, cost = sw._sa_seriate(D, init, seed=sa_seed + r,
                                         n_steps_multiplier=sa_mult)
            adj = sw.bifolio_adjacency_rate(order, doc_ids, folio_bifolio)
            results.append((adj, float(cost)))
        best_r = int(np.argmin([c for _, c in results]))

        for r, (adj, cost) in enumerate(results):
            rows.append({
                "source_text":        label,
                "window_idx":         0,
                "window_start":       0,
                "bifolios_per_quire": BPQ_NOT_APPLICABLE,
                "metric":             metric,
                "restart":            r,
                "init":               "svd" if r == 0 else "random",
                "n_folios":           n_docs,
                "n_quires":           n_quires,
                "n_bifolios":         info["n_bifolios"],
                "k_min_used":         k_min,
                "k_max_used":         k_max,
                "bifolio_adj_rate":   round(adj, 6),
                "random_baseline":    round(base_exact, 6),
                "random_baseline_foliation": round(base_fol, 6),
                "carob":              round(adj / base_exact, 6) if base_exact > 0 else float("nan"),
                "lift_over_baseline": round(adj - base_exact, 6),
                "seriation_cost":     round(cost, 6),
                "best_cost_restart":  r == best_r,
            })
        carobs = [r_["carob"] for r_ in rows[-restarts:]]
        print(f"done in {time.time() - t0:.1f}s  carob mean={np.nanmean(carobs):.3f} "
              f"max={np.nanmax(carobs):.3f}")

    info.update(k_min_used=k_min, k_max_used=k_max)
    return rows, info


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="Run the sweep_v2 pipeline on the VMS.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--foliation", type=Path, default=FOLIATION_CSV)
    ap.add_argument("--out", type=Path, default=OUTPUT_DIR)
    ap.add_argument("--label", default=SOURCE_LABEL, help="source_text label for the rows")
    ap.add_argument("--match-sweep", type=Path, default=None,
                    help="a sweep_v2 text output folder (or its config.json): copy its "
                         "restarts / nmf seeds / SA steps / K range / metrics")
    ap.add_argument("--metrics", nargs="+", choices=sorted(sw.DIST_FN), default=None)
    ap.add_argument("--restarts", type=int, default=None)
    ap.add_argument("--nmf-seeds", type=int, default=None)
    ap.add_argument("--sa-steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    # Settings precedence: explicit flag > --match-sweep > module defaults
    sweep_cfg = load_sweep_settings(args.match_sweep) if args.match_sweep else None

    def pick(flag, key, default):
        if flag is not None:
            return flag
        return sweep_cfg[key] if sweep_cfg else default

    metrics   = pick(args.metrics, "metrics", DISTANCE_METRICS)
    restarts  = pick(args.restarts, "restarts", N_RESTARTS)
    nmf_seeds = pick(args.nmf_seeds, "nmf_seeds", NMF_SEEDS)
    sa_mult   = pick(args.sa_steps, "sa_mult", SA_STEP_MULT)
    k_min, k_max = (sweep_cfg["k"] if sweep_cfg else (K_MIN, K_MAX))

    if not _VMS_PATTERN.search(args.label):
        print(f"  NOTE: label {args.label!r} does not match violin_v2's default "
              f"--vms-pattern ('voynich|vms'). That's fine with --vms-input, but if you "
              f"append these rows to results_full_all.csv they won't be recognised as VMS.")

    print("=" * 66)
    print("Voynich Manuscript — sweep_v2 pipeline on predetermined foliation")
    print(f"  Input  : {args.foliation}")
    print(f"  Label  : {args.label}")
    print(f"  Metrics: {metrics}")
    print(f"  NMF    : K={k_min}-{k_max}, {nmf_seeds} seeds")
    print(f"  SA     : {restarts} restarts, {sa_mult} steps/folio")
    if sweep_cfg:
        print(f"  Settings matched to sweep_v2 run: {args.match_sweep}")
    print("=" * 66)

    df = load_foliation(args.foliation)
    rows, info = run_vms(df, args.label, metrics, k_min, k_max,
                         nmf_seeds, restarts, sa_mult, args.seed)

    # ---- Window-length comparison (only if settings came from a sweep) ----
    if sweep_cfg and sweep_cfg.get("window_len"):
        wl = sweep_cfg["window_len"]
        diff = (info["n_tokens"] - wl) / wl
        msg = (f"  VMS has {info['n_tokens']} tokens; the sweep_v2 windows had {wl} words "
               f"({diff:+.1%}).")
        print(msg + ("  WARNING: lengths differ by >5%; CAROB varies with text length "
                     "(see carob_vs_length.py)." if abs(diff) > 0.05 else ""))

    # ---- Outputs ----
    args.out.mkdir(parents=True, exist_ok=True)
    full = pd.DataFrame(rows)
    full_path = args.out / "results_full.csv"
    full.to_csv(full_path, index=False)

    summary = sv2.summarise(full, ["metric"])
    summary_path = args.out / "results_summary.csv"
    summary.to_csv(summary_path, index=False)

    cfg_out = dict(
        script="voynich_run_v2.py", version=_SCRIPT_VERSION, source_text=args.label,
        foliation_csv=str(args.foliation), foliation_sha256=sha256_of(args.foliation),
        metrics=metrics, restarts=restarts, nmf_seeds=nmf_seeds, sa_mult=sa_mult,
        k=[k_min, k_max], seed=args.seed,
        matched_sweep=str(args.match_sweep) if args.match_sweep else None,
        bifolios_per_quire_sentinel=BPQ_NOT_APPLICABLE, **info)
    (args.out / "config.json").write_text(json.dumps(cfg_out, indent=2))

    # ---- Console summary ----
    print(f"\nFull results -> {full_path}  ({len(full)} rows)")
    print(f"Summary      -> {summary_path}")
    print(f"Config       -> {args.out / 'config.json'}")
    print("\n--- VMS CAROB by metric ---")
    print(summary[["metric", "carob_mean", "carob_restart_sd", "carob_mincost_mean",
                   "carob_max", "adj_rate_mean", "random_baseline_mean"]]
          .to_string(index=False))
    v_all = float(full["carob"].median())
    v_best = float(full.loc[full["best_cost_restart"], "carob"].median())
    print(f"\nVMS value as violin_v2 will compute it (median over all rows): {v_all:.3f}")
    print(f"  ... with --best-cost-only: {v_best:.3f}")
    print("\nNext:")
    print(f"  python violin_v2.py --input <outputs_v2>/results_full_all.csv "
          f"--vms-input {full_path}")
    print("  Note: violin_v2's --wpf filter leaves these rows alone (no "
          "words_per_folio value); --metrics and --best-cost-only apply to them.")
    print("\nDone.")


if __name__ == "__main__":
    main()
