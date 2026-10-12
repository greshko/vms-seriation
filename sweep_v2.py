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
sweep_v2.py
===========
Bootstrapped parameter sweep of the "conjoint adjacency ratio over baseline"
(CAROB) for synthetic foliations of one or more texts.

    CAROB = bifolio_adj_rate / random_baseline

1.0 means the recovered ordering is no better than a random ordering at
placing bifolio-mates next to each other; larger is better.

What changed relative to sweep.py
---------------------------------
* CAROB is the headline metric (ratio, not difference).
* The random baseline is CORRECTED. foliation.random_baseline() uses 2/(n-1)
  as the chance that two given folios are adjacent in a random ordering and
  then divides by the n-1 adjacent slots. For a linear ordering the chance is
  2/n, so the true expected adjacency rate is

        E[rate] = 2 * P / (n * (n - 1))

  where P = number of bifolio-mate pairs (sum over bifolios of C(size, 2)) and
  n = number of folios. (Checked against brute-force shuffles.) The old value
  is kept in the output as random_baseline_foliation for comparison.
* Bootstrapping: from each input text, N_WINDOWS contiguous samples of
  WINDOW_LEN words (default 10 x 38,525) are drawn. Start offsets are
  stratified: the valid start range is split into N_WINDOWS equal bins and one
  start is drawn at random per bin, so windows are spread over the text. If a
  text is only slightly longer than WINDOW_LEN, windows necessarily overlap
  heavily (the overlap is reported). Texts shorter than WINDOW_LEN are skipped.
* Input can be a folder (every *.txt in it) or a single .txt file.
* Same grid as sweep.py: words_per_folio x bifolios_per_quire x metric x
  initialization (SA restarts; restart 0 starts from the SVD order, the others
  from random permutations).
* Efficiency, without changing the method: folio contents do not depend on
  bifolios_per_quire, and the NMF ensemble does not depend on the metric. So
  for each (window, words_per_folio) the NMF ensemble is fitted once; for each
  metric the distance matrix and the SA restarts are run once; and every
  resulting ordering is then scored against every bifolios_per_quire. This
  also makes the quire-construction comparison paired (same orderings, only
  the scoring structure differs). Pairwise distances are vectorised with numpy
  (verified equal to sweep.py's loops; run with --selftest).
* Seeding uses numpy SeedSequence rather than hash(), so runs are reproducible
  across processes, and is independent of the number of workers.
* Resumable: each (text, window, words_per_folio) task is saved as it
  finishes. Re-running the same command skips finished tasks.

Outputs (OUTPUT_DIR/<stem>/ per text, and OUTPUT_DIR/ for all texts)
-------
  results_full.csv / results_full_all.csv
        one row per (window, wpf, bpq, metric, restart)
  results_summary.csv / results_summary_all.csv
        grid cell summaries (see summarise() for column definitions)
  results_summary_pooled.csv   all texts pooled (windows from every text)
  carob_viz.png / carob_viz_all.png   (needs matplotlib; skipped if absent)

Usage
-----
    python sweep_v2.py texts/                 # every .txt in the folder
    python sweep_v2.py texts/mytext.txt
    python sweep_v2.py texts/ --workers 8 --windows 10 --window-len 38525
    python sweep_v2.py --selftest

Requires foliation.py and sweep.py in the same directory.
"""

from __future__ import annotations

import os

# One BLAS thread per process; parallelism comes from the worker pool.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import math
import tempfile
import time
import warnings
import zlib
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import foliation as fol
import sweep as sw

# =============================================================================
# CONFIGURATION  (defaults; most can be overridden on the command line)
# =============================================================================

TEXTS_DIR  = Path("texts")
OUTPUT_DIR = Path("outputs_v2")

# Bootstrap
WINDOW_LEN = 38_525        # words per contiguous sample
N_WINDOWS  = 10            # samples per text

# Parameter grid (defaults inherited from sweep.py)
WORDS_PER_FOLIO_VALUES    = list(sw.WORDS_PER_FOLIO_VALUES)
BIFOLIOS_PER_QUIRE_VALUES = list(sw.BIFOLIOS_PER_QUIRE_VALUES)
DISTANCE_METRICS          = list(sw.DISTANCE_METRICS)

# NMF / seriation (inherited from sweep.py)
K_MIN, K_MAX, NMF_SEEDS = sw.K_MIN, sw.K_MAX, sw.NMF_SEEDS
N_RESTARTS   = sw.N_RESTARTS
SA_STEP_MULT = 200

GLOBAL_SEED = sw.GLOBAL_SEED
_EPS = 1e-12
_SCRIPT_VERSION = 2


# =============================================================================
# CORRECT RANDOM BASELINE
# =============================================================================

def exact_random_baseline(bifolio_of_folio: dict[int, int]) -> float:
    """
    Expected fraction of the n-1 consecutive pairs in a uniformly random
    linear ordering of n folios that are bifolio-mates.

    A given pair of folios is adjacent with probability 2/n, so the expected
    number of adjacent mate pairs is P * 2/n; dividing by the n-1 consecutive
    pairs gives 2P / (n (n-1)).
    """
    n = len(bifolio_of_folio)
    if n < 2:
        return 0.0
    sizes = Counter(bifolio_of_folio.values())
    P = sum(s * (s - 1) // 2 for s in sizes.values())
    return 2.0 * P / (n * (n - 1))


# =============================================================================
# VECTORISED DISTANCES (equivalent to sweep._hellinger / _js / _l1)
# =============================================================================

def pairwise_distances(W: np.ndarray, metric: str) -> np.ndarray:
    """(n, K) topic matrix -> (n, n) distance matrix."""
    P = np.asarray(W, dtype=np.float64)
    if metric == "hellinger":
        S = np.sqrt(np.clip(P, 0.0, 1.0))
        D = np.linalg.norm(S[:, None, :] - S[None, :, :], axis=2) / math.sqrt(2.0)
    elif metric == "l1":
        D = 0.5 * np.abs(P[:, None, :] - P[None, :, :]).sum(axis=2)
    elif metric == "js":
        Pc = np.clip(P, _EPS, 1.0)
        M = np.clip(0.5 * (P[:, None, :] + P[None, :, :]), _EPS, 1.0)
        kl = (Pc[:, None, :] * np.log(Pc[:, None, :] / M)).sum(axis=2)  # KL(p_a || m_ab)
        D = np.sqrt(np.maximum(0.5 * (kl + kl.T), 0.0))
    else:
        raise ValueError(f"unknown metric {metric!r}")
    np.fill_diagonal(D, 0.0)
    return D


def weighted_distance_matrix(doc_topic_by_KS: dict, weights_by_KS: dict,
                             metric: str, n_docs: int) -> np.ndarray:
    """Weighted-average distance over the NMF ensemble (cf. sweep.build_distance_matrix)."""
    D = np.zeros((n_docs, n_docs), dtype=np.float64)
    for ks, w in weights_by_KS.items():
        D += w * pairwise_distances(doc_topic_by_KS[ks], metric)
    return D


# =============================================================================
# SAMPLING
# =============================================================================

def sample_starts(n_tokens: int, length: int, n_windows: int,
                  rng: np.random.Generator) -> list[int]:
    """
    Up to n_windows distinct start offsets for contiguous `length`-word
    windows. Stratified: [0, n_tokens-length] is split into equal bins and one
    start is drawn at random from each.
    """
    max_start = n_tokens - length
    if max_start < 0:
        return []
    if max_start == 0:
        return [0]
    k = min(n_windows, max_start + 1)
    edges = np.linspace(0, max_start + 1, k + 1)
    starts = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        lo_i = int(np.ceil(lo))
        hi_i = min(max(int(np.ceil(hi)), lo_i + 1), max_start + 1)
        starts.append(int(rng.integers(lo_i, hi_i)))
    return sorted(set(starts))


# =============================================================================
# WORKER: one (text, window, words_per_folio) task
# =============================================================================

_CFG: dict | None = None


def _init_worker(cfg: dict) -> None:
    global _CFG
    _CFG = cfg


def _run_task(task: tuple) -> tuple[tuple, list[dict], float, str]:
    """
    Returns (task_key, rows, seconds, note). `rows` holds one dict per
    (bpq, metric, restart).
    """
    stem, w_idx, start, wpf, tokens = task
    key = (stem, w_idx, start, wpf)
    cfg = _CFG
    t0 = time.time()

    st = np.random.SeedSequence(
        [cfg["seed"], zlib.crc32(stem.encode()), start, wpf]).generate_state(4)
    nmf_seed = int(st[0] % 2**31)
    init_rng = np.random.default_rng(int(st[1]))
    sw.rng = np.random.default_rng(int(st[2]))        # used by sw._auto_temperature
    sa_seed = int(st[3] % 2**31)

    # --- 1. Foliations: one per bpq, via foliation.generate_foliation ---------
    folio_info: dict[int, dict] = {}
    counts_full = None
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "window.txt"
        p.write_text(" ".join(tokens), encoding="utf-8")
        for bpq in cfg["bpqs"]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                df = fol.generate_foliation(p, wpf, bpq)
            summ = fol.foliation_summary(df)
            fb = (df[["doc_id", "bifolio_global"]].drop_duplicates("doc_id")
                  .set_index("doc_id")["bifolio_global"].to_dict())
            folio_info[bpq] = dict(
                folio_bifolio=fb,
                n_quires=summ["n_quires"], n_bifolios=summ["n_bifolios"],
                base_exact=exact_random_baseline(fb),
                base_foliation=fol.random_baseline(df),
            )
            if counts_full is None:      # folio contents don't depend on bpq
                counts_full = (df.groupby(["doc_id", "token"]).size()
                                 .unstack(fill_value=0).sort_index())

    # --- 2. Counts + pruning --------------------------------------------------
    counts = sw.prune_vocabulary(counts_full)
    doc_ids = counts.index.tolist()
    n_docs = len(doc_ids)
    if counts.shape[1] == 0 or n_docs < 3:
        return key, [], time.time() - t0, "skipped (no vocabulary or <3 folios)"

    # --- 3. NMF ensemble (once; independent of metric and bpq) ---------------
    k_max = min(cfg["k_max"], n_docs, counts.shape[1])
    k_min = min(cfg["k_min"], k_max)
    dt, _errs, wts = sw.build_nmf_ensemble(
        counts, k_min, k_max, cfg["nmf_seeds"], base_seed=nmf_seed)
    svd_order = sw._svd_initial_order(dt, wts, n_docs, k_max)
    inits = [svd_order[:]] + [
        [int(x) for x in init_rng.permutation(n_docs)]
        for _ in range(cfg["restarts"] - 1)
    ]

    # --- 4. Per metric: distances + SA restarts; score against every bpq ------
    rows: list[dict] = []
    for metric in cfg["metrics"]:
        D = weighted_distance_matrix(dt, wts, metric, n_docs)
        results = []
        for r, init in enumerate(inits):
            order, cost = sw._sa_seriate(
                D, init, seed=sa_seed + r, n_steps_multiplier=cfg["sa_mult"])
            results.append((order, float(cost)))
        best_r = int(np.argmin([c for _, c in results]))

        for bpq in cfg["bpqs"]:
            info = folio_info[bpq]
            base = info["base_exact"]
            for r, (order, cost) in enumerate(results):
                adj = sw.bifolio_adjacency_rate(order, doc_ids, info["folio_bifolio"])
                rows.append({
                    "source_text":        stem,
                    "window_idx":         w_idx,
                    "window_start":       start,
                    "words_per_folio":    wpf,
                    "bifolios_per_quire": bpq,
                    "metric":             metric,
                    "restart":            r,
                    "init":               "svd" if r == 0 else "random",
                    "n_folios":           n_docs,
                    "n_quires":           info["n_quires"],
                    "n_bifolios":         info["n_bifolios"],
                    "k_min_used":         k_min,
                    "k_max_used":         k_max,
                    "bifolio_adj_rate":   round(adj, 6),
                    "random_baseline":    round(base, 6),
                    "random_baseline_foliation": round(info["base_foliation"], 6),
                    "carob":              round(adj / base, 6) if base > 0 else float("nan"),
                    "lift_over_baseline": round(adj - base, 6),
                    "seriation_cost":     round(cost, 6),
                    "best_cost_restart":  r == best_r,
                })
    return key, rows, time.time() - t0, "ok"


# =============================================================================
# SUMMARISING
# =============================================================================

GRID = ["words_per_folio", "bifolios_per_quire", "metric"]


def summarise(full: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """
    Two-level summary. Restarts within a window are not independent samples of
    the text, so the headline spread is across WINDOWS.

      carob_mean          mean over windows of (mean over restarts)
      carob_window_sd     SD across windows of the per-window mean   <- text-sampling variability
      carob_restart_sd    mean over windows of the SD across restarts <- initialization variability
      carob_mincost_mean  mean over windows of CAROB at the lowest-seriation-cost
                          restart (chosen without looking at ground truth)
      carob_mincost_sd    SD across windows of that value
      carob_max           best single restart anywhere in the cell
      n_windows           windows contributing (across texts, if pooled)
    """
    wkey = by + [c for c in ("source_text", "window_idx") if c not in by]
    win = (full.groupby(wkey)
               .apply(lambda g: pd.Series({
                   "carob_win_mean":    g["carob"].mean(),
                   "carob_win_sd":      g["carob"].std(ddof=1),
                   "carob_win_best":    g.loc[g["best_cost_restart"], "carob"].mean(),
                   "adj_win_mean":      g["bifolio_adj_rate"].mean(),
                   "baseline":          g["random_baseline"].iloc[0],
                   "baseline_old":      g["random_baseline_foliation"].iloc[0],
                   "carob_win_max":     g["carob"].max(),
                   "n_folios":          g["n_folios"].iloc[0],
                   "n_quires":          g["n_quires"].iloc[0],
                   "n_bifolios":        g["n_bifolios"].iloc[0],
                   "n_restarts":        len(g),
               }), include_groups=False)
               .reset_index())
    out = (win.groupby(by)
              .agg(carob_mean=("carob_win_mean", "mean"),
                   carob_window_sd=("carob_win_mean", "std"),
                   carob_restart_sd=("carob_win_sd", "mean"),
                   carob_mincost_mean=("carob_win_best", "mean"),
                   carob_mincost_sd=("carob_win_best", "std"),
                   carob_max=("carob_win_max", "max"),
                   adj_rate_mean=("adj_win_mean", "mean"),
                   random_baseline_mean=("baseline", "mean"),
                   random_baseline_foliation_mean=("baseline_old", "mean"),
                   n_folios=("n_folios", "mean"),
                   n_quires=("n_quires", "mean"),
                   n_bifolios=("n_bifolios", "mean"),
                   n_windows=("carob_win_mean", "count"),
                   n_restarts=("n_restarts", "first"))
              .reset_index()
              .round(6))
    return out


def console_summary(summ: pd.DataFrame, label: str) -> None:
    print(f"\n--- {label}: mean CAROB (rows = bifolios/quire, cols = words/folio; "
          f"averaged over metrics) ---")
    print(summ.pivot_table(index="bifolios_per_quire", columns="words_per_folio",
                           values="carob_mean").round(3).to_string())
    print(f"\n--- {label}: mean CAROB by metric (averaged over wpf, bpq) ---")
    print(summ.groupby("metric")["carob_mean"].mean().round(3).to_string())


def plot_summary(summ: pd.DataFrame, out_path: Path, title: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed; skipping chart)")
        return False

    colors = {"hellinger": "#2563eb", "js": "#16a34a", "l1": "#dc2626"}
    wpfs = sorted(summ["words_per_folio"].unique())
    metrics = [m for m in DISTANCE_METRICS if m in set(summ["metric"])] or \
        sorted(summ["metric"].unique())

    fig, axes = plt.subplots(1, len(wpfs), figsize=(4.8 * len(wpfs), 4.8),
                             sharey=True, squeeze=False)
    for ax, wpf in zip(axes[0], wpfs):
        for k, m in enumerate(metrics):
            s = (summ[(summ["words_per_folio"] == wpf) & (summ["metric"] == m)]
                 .sort_values("bifolios_per_quire"))
            if s.empty:
                continue
            x = s["bifolios_per_quire"] + (k - (len(metrics) - 1) / 2) * 0.06
            ax.errorbar(x, s["carob_mean"], yerr=s["carob_window_sd"].fillna(0),
                        marker="o", ms=4, lw=1.8, capsize=3,
                        color=colors.get(m, None), label=m)
        ax.axhline(1.0, ls="--", color="#888", lw=1)
        ax.set_title(f"{wpf} words / folio", fontsize=10)
        ax.set_xlabel("Bifolios per quire")
        ax.set_xticks(sorted(summ["bifolios_per_quire"].unique()))
        ax.grid(alpha=0.25)
    axes[0][0].set_ylabel("CAROB (adjacency rate / random baseline)")
    axes[0][-1].legend(frameon=False, title="metric")
    fig.suptitle(title, fontsize=11)
    fig.text(0.01, 0.005,
             "points = mean over windows; bars = ±1 SD across windows; "
             "dashed = random ordering (CAROB = 1)", fontsize=8, color="#666")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


# =============================================================================
# SELF-TEST
# =============================================================================

def selftest() -> None:
    """Check vectorised distances against sweep.py's, and the baseline by brute force."""
    rs = np.random.default_rng(0)
    W = rs.dirichlet(np.full(7, 0.3), size=25)
    W[3, :3] = 0.0
    W[3] /= W[3].sum()                                   # include exact zeros
    for m, fn in sw.DIST_FN.items():
        ref = np.array([[0.0 if a == b else fn(W[a], W[b]) for b in range(25)]
                        for a in range(25)])
        err = np.abs(pairwise_distances(W, m) - ref).max()
        print(f"  distance {m:<9} max |vectorised - sweep.py| = {err:.2e}")
        assert err < 1e-9, m

    prs = np.random.default_rng(1)
    for n, bpq in [(9, 4), (25, 4), (12, 2), (30, 1)]:
        F = 2 * bpq
        fb = {}
        for f in range(1, n + 1):
            q, fi = (f - 1) // F + 1, (f - 1) % F + 1
            fb[f] = (q - 1) * bpq + min(fi, F - fi + 1)
        ids = list(fb)
        hits = []
        for _ in range(40000):
            perm = prs.permutation(ids)
            hits.append(np.mean([fb[perm[i]] == fb[perm[i + 1]] for i in range(n - 1)]))
        emp, ex = float(np.mean(hits)), exact_random_baseline(fb)
        print(f"  baseline n={n:>2} bpq={bpq}: empirical {emp:.4f}   exact {ex:.4f}")
        assert abs(emp - ex) < 0.004 + 0.02 * ex
    print("selftest OK")


# =============================================================================
# MAIN
# =============================================================================

def _discover(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.glob("*.txt"))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Bootstrapped CAROB sweep.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path, nargs="?", default=TEXTS_DIR,
                    help="folder of .txt files, or one .txt file")
    ap.add_argument("--window-len", type=int, default=WINDOW_LEN)
    ap.add_argument("--windows", type=int, default=N_WINDOWS)
    ap.add_argument("--wpf", type=int, nargs="+", default=WORDS_PER_FOLIO_VALUES)
    ap.add_argument("--bpq", type=int, nargs="+", default=BIFOLIOS_PER_QUIRE_VALUES)
    ap.add_argument("--metrics", nargs="+", choices=sorted(sw.DIST_FN),
                    default=DISTANCE_METRICS)
    ap.add_argument("--restarts", type=int, default=N_RESTARTS,
                    help="initializations per (window, wpf, metric); restart 0 = SVD order")
    ap.add_argument("--nmf-seeds", type=int, default=NMF_SEEDS)
    ap.add_argument("--sa-steps", type=int, default=SA_STEP_MULT)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--seed", type=int, default=GLOBAL_SEED)
    ap.add_argument("--out", type=Path, default=OUTPUT_DIR)
    ap.add_argument("--force", action="store_true",
                    help="discard saved task results and recompute")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    text_files = _discover(args.path)
    if not text_files:
        raise SystemExit(f"No .txt files found at {args.path}")

    cfg = dict(seed=args.seed, bpqs=args.bpq, metrics=args.metrics,
               k_min=K_MIN, k_max=K_MAX, nmf_seeds=args.nmf_seeds,
               restarts=args.restarts, sa_mult=args.sa_steps)

    # ------------------------------------------------------------------ plan
    print("=" * 70)
    print("CAROB sweep (v2)")
    print(f"  input         : {args.path}  ({len(text_files)} file(s))")
    print(f"  bootstrap     : {args.windows} contiguous windows x {args.window_len} words")
    print(f"  grid          : wpf={args.wpf}  bpq={args.bpq}  metrics={args.metrics}")
    print(f"  initializations: {args.restarts} SA restarts (restart 0 = SVD init)")
    print(f"  NMF           : K={K_MIN}-{K_MAX} x{args.nmf_seeds} seeds")
    print("=" * 70)

    plans: dict[str, dict] = {}
    all_tasks: list[tuple] = []
    for tf in text_files:
        stem = tf.stem
        tokens = fol.tokenize_clean(tf.read_text(encoding="utf-8"))
        n = len(tokens)
        if n < args.window_len:
            print(f"  [{stem}] {n} words < window length {args.window_len}: SKIPPED")
            continue
        rng = np.random.default_rng(
            np.random.SeedSequence([args.seed, zlib.crc32(stem.encode()), n]))
        starts = sample_starts(n, args.window_len, args.windows, rng)
        span = starts[-1] - starts[0]
        min_overlap = max(0.0, 1 - span / args.window_len)
        note = ""
        if len(starts) < args.windows:
            note += f"  only {len(starts)} distinct windows possible!"
        if min_overlap > 0:
            note += f"  (all windows overlap by >= {min_overlap:.0%})"
        print(f"  [{stem}] {n} words -> {len(starts)} windows, "
              f"starts {starts[0]}..{starts[-1]}{note}")

        tdir = args.out / stem
        (tdir / "tasks").mkdir(parents=True, exist_ok=True)
        sig = json.loads(json.dumps(dict(
            version=_SCRIPT_VERSION, n_tokens=n, starts=starts,
            window_len=args.window_len, wpf=args.wpf, bpq=args.bpq,
            metrics=args.metrics, restarts=args.restarts,
            nmf_seeds=args.nmf_seeds, k=[K_MIN, K_MAX],
            sa_mult=args.sa_steps, seed=args.seed)))
        cfg_path = tdir / "config.json"
        if cfg_path.exists() and not args.force:
            if json.loads(cfg_path.read_text()) != sig:
                raise SystemExit(
                    f"Saved results in {tdir} were made with different settings. "
                    f"Use --force to recompute, or choose another --out.")
        else:
            for f in (tdir / "tasks").glob("*.csv"):
                f.unlink()
        cfg_path.write_text(json.dumps(sig, indent=2))

        plans[stem] = dict(tdir=tdir, starts=starts, n=n, rows=[])
        for w_idx, s in enumerate(starts):
            window = tokens[s:s + args.window_len]
            for wpf in args.wpf:
                all_tasks.append((stem, w_idx, s, wpf, window))

    if not plans:
        raise SystemExit("Nothing to do: no text is long enough.")

    def _task_file(stem, w_idx, s, wpf) -> Path:
        return plans[stem]["tdir"] / "tasks" / f"w{w_idx:02d}_s{s}_wpf{wpf}.csv"

    todo = []
    for t in all_tasks:
        f = _task_file(*t[:4])
        if f.exists():
            plans[t[0]]["rows"].extend(pd.read_csv(f).to_dict("records"))
        else:
            todo.append(t)
    todo.sort(key=lambda t: t[3])              # small wpf (most folios) first
    print(f"\n  tasks: {len(all_tasks)} total, {len(all_tasks) - len(todo)} already "
          f"done, {len(todo)} to run  (workers={args.workers})\n", flush=True)

    # ------------------------------------------------------------------- run
    t_start = time.time()

    def _handle(i: int, res: tuple) -> None:
        key, rows, secs, note = res
        stem, w_idx, s, wpf = key
        if rows:
            pd.DataFrame(rows).to_csv(_task_file(*key), index=False)
            plans[stem]["rows"].extend(rows)
        el = time.time() - t_start
        eta = el / i * (len(todo) - i)
        c = [r["carob"] for r in rows if r["metric"] == args.metrics[0]
             and r["bifolios_per_quire"] == args.bpq[0]]
        c_txt = f"carob~{np.nanmean(c):.2f}" if c else note
        print(f"  [{i}/{len(todo)}] {stem} w{w_idx} start={s} wpf={wpf} "
              f"({secs:.0f}s)  {c_txt}   elapsed {el/60:.1f}m  eta {eta/60:.1f}m",
              flush=True)

    if args.workers <= 1:
        _init_worker(cfg)
        for i, t in enumerate(todo, 1):
            _handle(i, _run_task(t))
    elif todo:
        with Pool(args.workers, initializer=_init_worker, initargs=(cfg,)) as pool:
            for i, res in enumerate(pool.imap_unordered(_run_task, todo, chunksize=1), 1):
                _handle(i, res)

    # --------------------------------------------------------------- outputs
    per_text_full = []
    for stem, plan in plans.items():
        if not plan["rows"]:
            print(f"\n  [{stem}] no results.")
            continue
        full = (pd.DataFrame(plan["rows"])
                  .sort_values(["window_idx", "words_per_folio", "bifolios_per_quire",
                                "metric", "restart"]).reset_index(drop=True))
        per_text_full.append(full)
        tdir = plan["tdir"]
        full.to_csv(tdir / "results_full.csv", index=False)
        summ = summarise(full, GRID)
        summ.to_csv(tdir / "results_summary.csv", index=False)
        plot_summary(summ, tdir / "carob_viz.png",
                     f"CAROB — {stem}  ({summ['n_windows'].max()} windows x "
                     f"{args.window_len} words)")
        print(f"\n  [{stem}] -> {tdir}/results_full.csv ({len(full)} rows), "
              f"results_summary.csv, carob_viz.png")
        console_summary(summ, stem)

    if not per_text_full:
        raise SystemExit("No results generated.")

    args.out.mkdir(parents=True, exist_ok=True)
    all_full = pd.concat(per_text_full, ignore_index=True)
    all_full.to_csv(args.out / "results_full_all.csv", index=False)
    summarise(all_full, ["source_text"] + GRID).to_csv(
        args.out / "results_summary_all.csv", index=False)
    pooled = summarise(all_full, GRID)
    pooled.to_csv(args.out / "results_summary_pooled.csv", index=False)
    plot_summary(pooled, args.out / "carob_viz_all.png",
                 f"CAROB — all texts pooled ({len(per_text_full)} text(s), "
                 f"{pooled['n_windows'].max()} windows)")
    print(f"\n{'=' * 70}")
    print(f"Combined -> {args.out}/results_full_all.csv ({len(all_full)} rows), "
          f"results_summary_all.csv, results_summary_pooled.csv, carob_viz_all.png")
    if len(per_text_full) > 1:
        console_summary(pooled, "ALL TEXTS (pooled)")
    print(f"\nDone in {(time.time() - t_start) / 60:.1f} min.")


if __name__ == "__main__":
    main()
