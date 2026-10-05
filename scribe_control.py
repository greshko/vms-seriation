"""
scribe_control.py
=================
Paired control for the "scribe-per-bifolium" confound.

Seriation sees only folio contents, never the binding or pairing. So we fit the
pipeline ONCE per (window, wpf) on a five-text miscellany cut into folio-sized
chunks, then score every recovered ordering against several definitions of
which folia are conjoint ("pairings"):

  nested_bpq1..5   : the paper's design (leaves consecutive / nested in quires)
  src_w{W}         : each leaf paired with another leaf from the SAME source
                     text, at random, non-adjacent, within W chunks (W=inf: anywhere)
  src_w1           : same-source pairing at distance exactly 1 (sanity check; ~singulion)

Uses sweep.py / sweep_v2.py / foliation.py functions.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, json, time, zlib
from pathlib import Path
import numpy as np
import pandas as pd

import foliation as fol
import sweep as sw
import sweep_v2 as sv2

HERE = Path(__file__).parent
SOURCES = [  # (stem, file, whole text or section length)
    ("Petrus",    "texts/miscellany-sources/CC_553_Petrus-de-Ebulo_De-balneis-puteolanis.txt", None),
    ("Liber",     "texts/miscellany-sources/Liber_aggregationis.txt", None),
    ("Sacrobosco","texts/miscellany-sources/Johannes-Sacrobosco_De-sphaera.txt", None),
    ("Trotula",   "texts/miscellany-sources/Trotula.txt", None),
    ("Secretum",  "texts/miscellany-sources/Roger-Bacon_Seretum-secretorum.txt", 4910),   # contiguous section
]
WINDOW_CHOICES = [None, 1, 2, 4, 8, 16]      # W; None = anywhere within source
BLOCKS = [2, 4, 8, 12]
N_PAIRINGS = 100                              # random pairings averaged per ordering


def load_tokens():
    toks = {}
    for stem, f, sec in SOURCES:
        toks[stem] = fol.tokenize_clean((HERE / f).read_text(encoding="utf-8", errors="replace"))
    return toks


def build_chunks(tokens_by_src, secretum_start, wpf):
    """Return list of (source, chunk_tokens) in concatenated order; full chunks only."""
    chunks = []
    for stem, f, sec in SOURCES:
        t = tokens_by_src[stem]
        if sec is not None:
            t = t[secretum_start:secretum_start + sec]
        m = len(t) // wpf
        for i in range(m):
            chunks.append((stem, t[i * wpf:(i + 1) * wpf]))
    return chunks


def nested_pairing(n, bpq):
    F = 2 * bpq
    out = {}
    for f in range(1, n + 1):
        q, fi = (f - 1) // F + 1, (f - 1) % F + 1
        out[f] = (q - 1) * bpq + min(fi, F - fi + 1)
    return out


def source_pairing(src_of, W, rng, tries=30):
    """
    src_of: list source label per folio (1-based folio = index+1), chunk order
    within a source is the list order. Randomly pair leaves of the same source,
    non-adjacent (distance >=2) and within W chunks (W=None -> unlimited);
    W == 1 means distance exactly 1. Unpairable leaves become singletons.
    Returns dict folio -> bifolio id (best of `tries` greedy draws).
    """
    n = len(src_of)
    by_src = {}
    for i, s in enumerate(src_of):
        by_src.setdefault(s, []).append(i)
    dmin = 1 if W == 1 else 2
    dmax = (n if W is None else W)
    best, best_single = None, 10 ** 9
    for _ in range(tries):
        mate = {}
        for s, idxs in by_src.items():
            pos = list(range(len(idxs)))
            free = set(pos)
            order = list(rng.permutation(pos))
            for c in order:
                if c not in free:
                    continue
                cands = [j for j in free if j != c and dmin <= abs(j - c) <= dmax]
                if not cands:
                    continue
                j = cands[int(rng.integers(len(cands)))]
                free.discard(c); free.discard(j)
                mate[idxs[c]] = idxs[j]; mate[idxs[j]] = idxs[c]
        singles = n - len(mate)
        if singles < best_single:
            best_single, best = singles, dict(mate)
            if singles <= (len(by_src) and sum(len(v) % 2 for v in by_src.values())):
                break
    bif, nxt = {}, 0
    for i in range(n):
        if i in bif:
            continue
        bif[i] = nxt
        if i in best:
            bif[best[i]] = nxt
        nxt += 1
    return {i + 1: b for i, b in bif.items()}, best_single


def block_pairing(src_of, B, rng):
    """Partition each source's chunks into consecutive blocks of B leaves; pair
    leaves at random inside each block (any distance). Odd leaf -> singleton."""
    n = len(src_of)
    by_src = {}
    for i, s in enumerate(src_of):
        by_src.setdefault(s, []).append(i)
    bif, nxt = {}, 0
    for s, idxs in by_src.items():
        for b0 in range(0, len(idxs), B):
            blk = [idxs[j] for j in range(b0, min(b0 + B, len(idxs)))]
            blk = [blk[j] for j in rng.permutation(len(blk))]
            for k in range(0, len(blk) - 1, 2):
                bif[blk[k]] = bif[blk[k + 1]] = nxt; nxt += 1
            if len(blk) % 2:
                bif[blk[-1]] = nxt; nxt += 1
    return {i + 1: b for i, b in bif.items()}


def run_task(w_idx, start, wpf, cfg, tokens_by_src):
    t0 = time.time()
    chunks = build_chunks(tokens_by_src, start, wpf)
    n = len(chunks)
    src_of = [c[0] for c in chunks]
    # counts matrix: one row per folio (doc_id 1..n)
    rows = []
    for f, (s, t) in enumerate(chunks, start=1):
        for w in t:
            rows.append((f, w))
    df = pd.DataFrame(rows, columns=["doc_id", "token"])
    counts_full = df.groupby(["doc_id", "token"]).size().unstack(fill_value=0).sort_index()
    counts = sw.prune_vocabulary(counts_full)
    doc_ids = counts.index.tolist()
    n_docs = len(doc_ids)

    st = np.random.SeedSequence([cfg["seed"], start, wpf]).generate_state(4)
    sw.rng = np.random.default_rng(int(st[2]))
    init_rng = np.random.default_rng(int(st[1]))
    dt, _e, wts = sw.build_nmf_ensemble(counts, sw.K_MIN, sw.K_MAX, sw.NMF_SEEDS,
                                        base_seed=int(st[0] % 2**31))
    svd_order = sw._svd_initial_order(dt, wts, n_docs, sw.K_MAX)
    inits = [svd_order[:]] + [[int(x) for x in init_rng.permutation(n_docs)]
                              for _ in range(cfg["restarts"] - 1)]
    # pairings
    prng = np.random.default_rng(int(st[3]))
    pairings = {f"nested_bpq{b}": [nested_pairing(n, b)] for b in (1, 2, 3, 4, 5)}
    singles = {}
    for W in WINDOW_CHOICES:
        name = "src_w" + ("inf" if W is None else str(W))
        lst, sing = [], []
        for _ in range(N_PAIRINGS):
            p, s = source_pairing(src_of, W, prng)
            lst.append(p); sing.append(s)
        pairings[name] = lst
        singles[name] = float(np.mean(sing))
    for B in BLOCKS:
        pairings[f"blk{B}"] = [block_pairing(src_of, B, prng) for _ in range(N_PAIRINGS)]
    base = {k: float(np.mean([sv2.exact_random_baseline(p) for p in v])) for k, v in pairings.items()}

    out = []
    for metric in cfg["metrics"]:
        D = sv2.weighted_distance_matrix(dt, wts, metric, n_docs)
        res = []
        for r, init in enumerate(inits):
            order, cost = sw._sa_seriate(D, init, seed=int(st[3] % 2**31) + r, n_steps_multiplier=200)
            res.append((order, float(cost)))
        best_r = int(np.argmin([c for _, c in res]))
        for r, (order, cost) in enumerate(res):
            for name, lst in pairings.items():
                cs = []
                for p in lst:
                    adj = sw.bifolio_adjacency_rate(order, doc_ids, p)
                    b = sv2.exact_random_baseline(p)
                    cs.append(adj / b if b > 0 else np.nan)
                out.append(dict(window_idx=w_idx, secretum_start=start, wpf=wpf, metric=metric,
                                restart=r, best_cost_restart=(r == best_r), pairing=name,
                                carob=float(np.nanmean(cs)), carob_pairing_sd=float(np.nanstd(cs)),
                                baseline=base[name], n_folios=n_docs,
                                mean_singletons=singles.get(name, 0.0), cost=cost))
    return out, time.time() - t0


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=5)
    ap.add_argument("--wpf", type=int, nargs="+", default=[378])
    ap.add_argument("--restarts", type=int, default=8)
    ap.add_argument("--metrics", nargs="+", default=["hellinger", "js", "l1"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=HERE / "out")
    a = ap.parse_args()
    a.out.mkdir(exist_ok=True, parents=True)
    cfg = dict(seed=a.seed, restarts=a.restarts, metrics=a.metrics)
    toks = load_tokens()
    nsec = len(toks["Secretum"])
    rng = np.random.default_rng(a.seed)
    starts = sv2.sample_starts(nsec, 4910, a.windows, rng)
    print("tokens:", {k: len(v) for k, v in toks.items()}, "| secretum starts:", starts, flush=True)
    tasks = [(i, s, w) for i, s in enumerate(starts) for w in a.wpf]
    t_all = time.time()
    for k, (i, s, w) in enumerate(tasks, 1):
        f = a.out / f"task_w{i}_s{s}_wpf{w}.csv"
        if f.exists():
            continue
        rows, secs = run_task(i, s, w, cfg, toks)
        pd.DataFrame(rows).to_csv(f, index=False)
        print(f"[{k}/{len(tasks)}] w{i} start={s} wpf={w} {secs:.0f}s  elapsed {(time.time()-t_all)/60:.1f}m", flush=True)
    full = pd.concat([pd.read_csv(p) for p in sorted(a.out.glob("task_*.csv"))], ignore_index=True)
    full.to_csv(a.out / "results_full.csv", index=False)
    print("done")


if __name__ == "__main__":
    main()
