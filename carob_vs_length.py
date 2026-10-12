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
carob_vs_length.py
==================
How does the "conjoint adjacency ratio over baseline" (CAROB) depend on the
length of the input text and on the number of words per folio?

    CAROB = bifolio_adj_rate / random_baseline

(1.0 = no better than a random ordering; larger = more bifolio-mates recovered
as neighbours.)

For one input TXT file this script:
  1. Tokenizes the text (foliation.tokenize_clean) and determines the
     truncation lengths: 5000, 10000, 15000, ... up to the largest multiple
     of LENGTH_STEP that does not exceed the number of words in the text.
  2. For each length L, draws up to N_WINDOWS contiguous L-word windows from
     the text (bootstrap by resampling). Window 0 is always the literal
     prefix truncation (words 0..L-1); the others are start offsets drawn at
     random, one per equal-width bin of the valid start range, so that they
     are spread across the text. Windows at large L necessarily overlap
     heavily; at the largest L there may be only one or a few distinct starts.
  3. For each (window, words_per_folio) runs the same pipeline as sweep.py
     (foliation -> prune vocabulary -> NMF ensemble -> distance matrix ->
     SA seriation with restarts -> bifolio adjacency rate), re-using sweep.py's
     functions directly.
  4. Writes per-window and summary CSVs and a PNG chart of CAROB vs text
     length, one line per words_per_folio value (mean across windows,
     shaded +/- 1 SD, individual windows as faint points).

bifolios_per_quire and the distance metric are held fixed within a run (set
BIFOLIOS_PER_QUIRE / METRIC below or use --bpq / --metric).

Usage
-----
    python carob_vs_length.py texts/mytext.txt
    python carob_vs_length.py texts/mytext.txt --windows 20 --workers 8
    python carob_vs_length.py texts/mytext.txt --baseline exact

Requires foliation.py and sweep.py in the same directory, plus matplotlib.
"""

from __future__ import annotations

import os

# One BLAS thread per process; parallelism comes from the worker pool.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import tempfile
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import foliation as fol
import sweep as sw

# =============================================================================
# CONFIGURATION  (defaults; most can be overridden on the command line)
# =============================================================================

LENGTH_STEP            = 5000                              # truncation increment (words)
WORDS_PER_FOLIO_VALUES = list(sw.WORDS_PER_FOLIO_VALUES)   # one chart line each
BIFOLIOS_PER_QUIRE     = 4                                 # held fixed in a run
METRIC                 = "hellinger"                       # held fixed in a run

N_WINDOWS   = 10                  # windows per length (incl. the prefix window)
N_RESTARTS  = sw.N_RESTARTS       # SA restarts per window
NMF_SEEDS   = sw.NMF_SEEDS
K_MIN       = sw.K_MIN
K_MAX       = sw.K_MAX
SA_STEP_MULT = 200                # SA steps = n_folios * this (as in sweep.py)

# Which restart defines a window's adjacency rate:
#   "min_cost" : the restart with the lowest seriation cost (unsupervised --
#                does not look at the ground-truth bifolios when choosing)
#   "mean"     : mean adjacency rate over restarts
RESTART_AGG = "min_cost"

# Which denominator to use for CAROB:
#   "foliation" : fol.random_baseline(), exactly as sweep.py reports it
#   "exact"     : the same value times (n-1)/n, i.e. the true expected
#                 adjacency rate of a uniformly random ordering (see note in
#                 the docstring of exact_baseline()).
BASELINE = "foliation"

SEED       = sw.GLOBAL_SEED
OUTPUT_DIR = Path("outputs") / "carob_vs_length"


# =============================================================================
# HELPERS
# =============================================================================

def exact_baseline(foliation_baseline: float, n_folios: int) -> float:
    """
    Correct fol.random_baseline() for its n/(n-1) overestimate.

    foliation.random_baseline uses P * 2/(n-1) / (n-1) for P bifolio-mate
    pairs among n folios. The probability that a given pair of folios is
    adjacent in a random permutation is 2/n (not 2/(n-1)), so the true
    expected rate is P * 2 / (n * (n-1)) = foliation_baseline * (n-1)/n.
    The difference is negligible for large n but ~12% at n = 9 folios.
    """
    if n_folios < 2:
        return 0.0
    return foliation_baseline * (n_folios - 1) / n_folios


def truncation_lengths(n_tokens: int, step: int) -> list[int]:
    """step, 2*step, ... largest multiple of step that is <= n_tokens."""
    return list(range(step, (n_tokens // step) * step + 1, step))


def sample_starts(n_tokens: int, length: int, n_windows: int,
                  rng: np.random.Generator) -> list[int]:
    """
    Choose up to n_windows distinct start offsets for contiguous windows of
    `length` words. Stratified: the valid range [0, n_tokens-length] is split
    into equal bins and one start is drawn per bin. The first bin is pinned to
    start=0 (the literal prefix truncation).
    """
    max_start = n_tokens - length
    if max_start <= 0:
        return [0]
    k = min(n_windows, max_start + 1)
    edges = np.linspace(0, max_start + 1, k + 1)
    starts = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        lo_i = int(np.ceil(lo))
        hi_i = max(int(np.ceil(hi)), lo_i + 1)
        hi_i = min(hi_i, max_start + 1)
        starts.append(int(rng.integers(lo_i, hi_i)))
    starts[0] = 0
    return sorted(set(starts))


# =============================================================================
# WORKER: one (length, start, words_per_folio) task
# =============================================================================

_TOKENS: list[str] | None = None
_CFG: dict | None = None


def _init_worker(tokens: list[str], cfg: dict) -> None:
    global _TOKENS, _CFG
    _TOKENS, _CFG = tokens, cfg


def _run_task(task: tuple[int, int, int]) -> dict:
    length, start, wpf = task
    cfg = _CFG
    t0 = time.time()

    # Deterministic, per-task seeding (independent of worker assignment).
    state = np.random.SeedSequence([cfg["seed"], length, start, wpf]).generate_state(3)
    base_seed = int(state[0] % 2**31)
    local_rng = np.random.default_rng(int(state[1]))
    sw.rng = np.random.default_rng(int(state[2]))   # used by sw._auto_temperature

    row = {
        "length": length, "start": start, "words_per_folio": wpf,
        "bifolios_per_quire": cfg["bpq"], "metric": cfg["metric"],
        "status": "ok",
    }

    # --- 1. Foliation of this window (via foliation.py's public API) --------
    window = _TOKENS[start:start + length]
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "window.txt"
        p.write_text(" ".join(window), encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            df = fol.generate_foliation(p, wpf, cfg["bpq"])

    summ = fol.foliation_summary(df)
    base_fol = fol.random_baseline(df)
    folio_meta = (df[["doc_id", "bifolio_global"]]
                  .drop_duplicates("doc_id")
                  .set_index("doc_id")["bifolio_global"].to_dict())
    row.update(n_folios=summ["n_folios"], n_bifolios=summ["n_bifolios"],
               n_quires=summ["n_quires"])

    # --- 2. Counts + vocabulary pruning -------------------------------------
    counts = sw.prune_vocabulary(
        df.groupby(["doc_id", "token"]).size().unstack(fill_value=0).sort_index()
    )
    doc_ids = counts.index.tolist()
    n_docs = len(doc_ids)
    if counts.shape[1] == 0 or n_docs < 3:
        row["status"] = "skipped (no vocabulary or <3 folios)"
        return row

    # --- 3. NMF ensemble, distances, seriation ------------------------------
    # NMF (nndsvda) requires K <= min(n_folios, n_vocab). Short windows with
    # large folios (e.g. 5k words / 600 wpf = 9 folios) can't support K=15, so
    # K is capped; the K range actually used is recorded in the output.
    k_max = min(cfg["k_max"], n_docs, counts.shape[1])
    k_min = min(cfg["k_min"], k_max)
    row.update(k_min_used=k_min, k_max_used=k_max)
    dt, _errs, wts = sw.build_nmf_ensemble(
        counts, k_min, k_max, cfg["nmf_seeds"], base_seed=base_seed)
    D = sw.build_distance_matrix(dt, wts, cfg["metric"], n_docs)
    svd_order = sw._svd_initial_order(dt, wts, n_docs, k_max)

    rates, costs = [], []
    for r in range(cfg["restarts"]):
        init = (svd_order[:] if r == 0
                else [int(x) for x in local_rng.permutation(n_docs)])
        order, cost = sw._sa_seriate(
            D, init, seed=base_seed + r, n_steps_multiplier=cfg["sa_mult"])
        rates.append(sw.bifolio_adjacency_rate(order, doc_ids, folio_meta))
        costs.append(float(cost))

    adj_mincost = rates[int(np.argmin(costs))]
    adj_mean = float(np.mean(rates))
    adj = adj_mincost if cfg["restart_agg"] == "min_cost" else adj_mean

    base_exact = exact_baseline(base_fol, n_docs)
    base_used = base_exact if cfg["baseline"] == "exact" else base_fol

    def _ratio(a: float, b: float) -> float:
        return a / b if b > 0 else float("nan")

    row.update(
        adj_rate_mincost=round(adj_mincost, 6),
        adj_rate_mean=round(adj_mean, 6),
        adj_rate_max=round(max(rates), 6),
        random_baseline_foliation=round(base_fol, 6),
        random_baseline_exact=round(base_exact, 6),
        carob_foliation_baseline=round(_ratio(adj, base_fol), 6),
        carob_exact_baseline=round(_ratio(adj, base_exact), 6),
        carob=round(_ratio(adj, base_used), 6),
        seconds=round(time.time() - t0, 1),
    )
    return row


# =============================================================================
# SUMMARY + CHART
# =============================================================================

def summarise(df: pd.DataFrame, agg: str) -> pd.DataFrame:
    ok = df[df["status"] == "ok"]
    return (
        ok.groupby(["words_per_folio", "length"])
          .agg(carob_mean=("carob", "mean"),
               carob_sd=("carob", "std"),
               carob_min=("carob", "min"),
               carob_max=("carob", "max"),
               n_windows=("carob", "count"),
               n_folios=("n_folios", "first"),
               adj_rate_mean=("adj_rate_mincost" if agg == "min_cost"
                              else "adj_rate_mean", "mean"))
          .reset_index()
          .round(6)
    )


def plot_carob(df: pd.DataFrame, out_path: Path, title: str, step: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = df[(df["status"] == "ok")].dropna(subset=["carob"])
    wpfs = sorted(ok["words_per_folio"].unique())
    colors = ["#2563eb", "#16a34a", "#dc2626", "#9333ea", "#ea580c"]

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for k, wpf in enumerate(wpfs):
        c = colors[k % len(colors)]
        sub = ok[ok["words_per_folio"] == wpf]
        g = (sub.groupby("length")["carob"]
                .agg(["mean", "std", "count"]).reset_index())
        dx = (k - (len(wpfs) - 1) / 2) * step * 0.04   # nudge so points don't overlap
        ax.scatter(sub["length"] + dx, sub["carob"], s=12, alpha=0.25,
                   color=c, linewidths=0)
        sd = g["std"].fillna(0.0)
        ax.fill_between(g["length"], g["mean"] - sd, g["mean"] + sd,
                        color=c, alpha=0.12, linewidth=0)
        ax.plot(g["length"], g["mean"], "-o", color=c, lw=2, ms=5,
                label=f"{wpf} words / folio")

    ax.axhline(1.0, ls="--", color="#888", lw=1, label="random ordering (CAROB = 1)")
    ax.set_xlabel("Text length (words)")
    ax.set_ylabel("CAROB  (bifolio adjacency rate / random baseline)")
    ax.set_title(title, fontsize=11)
    ax.set_xticks(sorted(ok["length"].unique()))
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.text(0.01, 0.005,
             "line = mean over windows; band = ±1 SD; dots = individual windows",
             fontsize=8, color="#666")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("Usage")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", type=Path, help="input .txt file")
    ap.add_argument("--step", type=int, default=LENGTH_STEP)
    ap.add_argument("--wpf", type=int, nargs="+", default=WORDS_PER_FOLIO_VALUES)
    ap.add_argument("--bpq", type=int, default=BIFOLIOS_PER_QUIRE)
    ap.add_argument("--metric", choices=sorted(sw.DIST_FN), default=METRIC)
    ap.add_argument("--windows", type=int, default=N_WINDOWS)
    ap.add_argument("--restarts", type=int, default=N_RESTARTS)
    ap.add_argument("--nmf-seeds", type=int, default=NMF_SEEDS)
    ap.add_argument("--sa-steps", type=int, default=SA_STEP_MULT,
                    help="SA steps per folio (sweep.py uses 200)")
    ap.add_argument("--agg", choices=["min_cost", "mean"], default=RESTART_AGG)
    ap.add_argument("--baseline", choices=["foliation", "exact"], default=BASELINE)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", type=Path, default=None,
                    help=f"output dir (default {OUTPUT_DIR}/<text stem>)")
    args = ap.parse_args()

    # --- Tokenize & plan -----------------------------------------------------
    text = args.text.read_text(encoding="utf-8")
    tokens = fol.tokenize_clean(text)
    n_tok = len(tokens)
    lengths = truncation_lengths(n_tok, args.step)
    if not lengths:
        raise SystemExit(f"{args.text} has only {n_tok} usable words; "
                         f"need at least {args.step}.")

    plan_rng = np.random.default_rng(args.seed)
    starts_by_len = {L: sample_starts(n_tok, L, args.windows, plan_rng)
                     for L in lengths}

    tasks = [(L, s, wpf)
             for L in lengths for s in starts_by_len[L] for wpf in args.wpf]
    tasks.sort(key=lambda t: -t[0])          # biggest first: better load balance

    out_dir = args.out or (OUTPUT_DIR / args.text.stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 66)
    print(f"CAROB vs text length — {args.text.name}")
    print(f"  words (cleaned) : {n_tok}")
    print(f"  lengths         : {lengths[0]} .. {lengths[-1]} by {args.step} "
          f"({len(lengths)} lengths)")
    print(f"  windows/length  : " + ", ".join(
        f"{L // 1000}k:{len(starts_by_len[L])}" for L in lengths))
    print(f"  words/folio     : {args.wpf}   bpq={args.bpq}   metric={args.metric}")
    print(f"  NMF K={K_MIN}-{K_MAX} x{args.nmf_seeds} seeds   "
          f"SA restarts={args.restarts}   restart agg={args.agg}")
    print(f"  baseline        : {args.baseline}")
    print(f"  tasks           : {len(tasks)}   workers={args.workers}")
    print("=" * 66, flush=True)

    cfg = dict(seed=args.seed, bpq=args.bpq, metric=args.metric,
               k_min=K_MIN, k_max=K_MAX, nmf_seeds=args.nmf_seeds,
               restarts=args.restarts, sa_mult=args.sa_steps,
               restart_agg=args.agg, baseline=args.baseline)

    # --- Run -----------------------------------------------------------------
    rows: list[dict] = []
    t_start = time.time()

    def _report(i: int, r: dict) -> None:
        carob = r.get("carob")
        tag = f"carob={carob:.3f}" if isinstance(carob, float) and carob == carob \
            else r["status"]
        print(f"  [{i}/{len(tasks)}] L={r['length']:>6} start={r['start']:>6} "
              f"wpf={r['words_per_folio']:>3}  n_folios={r.get('n_folios', '?'):>4}  "
              f"{tag}  ({time.time() - t_start:.0f}s elapsed)", flush=True)

    if args.workers <= 1:
        _init_worker(tokens, cfg)
        for i, t in enumerate(tasks, 1):
            r = _run_task(t)
            rows.append(r)
            _report(i, r)
    else:
        with Pool(args.workers, initializer=_init_worker,
                  initargs=(tokens, cfg)) as pool:
            for i, r in enumerate(pool.imap_unordered(_run_task, tasks, chunksize=1), 1):
                rows.append(r)
                _report(i, r)

    # --- Outputs -------------------------------------------------------------
    res = (pd.DataFrame(rows)
             .sort_values(["words_per_folio", "length", "start"])
             .reset_index(drop=True))
    res_path = out_dir / "carob_windows.csv"
    res.to_csv(res_path, index=False)

    summ = summarise(res, args.agg)
    summ_path = out_dir / "carob_summary.csv"
    summ.to_csv(summ_path, index=False)

    if "k_max_used" in res:
        n_cap = int((res["k_max_used"] < K_MAX).sum())
        if n_cap:
            print(f"\nNOTE: K was capped below {K_MAX} for {n_cap} window(s) "
                  f"(fewer folios than K). Short-text / large-folio points are "
                  f"therefore not fitted with the same K range as long-text ones; "
                  f"see k_max_used in the CSV.")
    n_nan = int(res["carob"].isna().sum()) if "carob" in res else len(res)
    if n_nan:
        print(f"\nNOTE: {n_nan} window(s) have no CAROB (skipped, or random "
              f"baseline = 0 because no bifolio-mates fit in the window).")

    png_path = out_dir / "carob_vs_length.png"
    title = (f"CAROB vs text length — {args.text.stem}\n"
             f"bifolios/quire={args.bpq}, metric={args.metric}, "
             f"{args.windows} windows/length, baseline={args.baseline}")
    plot_carob(res, png_path, title, args.step)

    print(f"\nPer-window results -> {res_path}")
    print(f"Summary            -> {summ_path}")
    print(f"Chart              -> {png_path}")
    print(f"Total time: {time.time() - t_start:.0f}s\n")
    print(summ.pivot(index="length", columns="words_per_folio",
                     values="carob_mean").round(3).to_string())


if __name__ == "__main__":
    main()
