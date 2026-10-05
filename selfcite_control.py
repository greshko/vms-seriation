"""
selfcite_control.py
===================
Paired CAROB control on a *meaningless* self-citation-generated text
(selfcite_text.txt; '#' metadata header trimmed).

Same design as scribe_control.py: the seriation sees only folio contents, so we
fit the pipeline once per (wpf, seed) on the text cut into folio-sized chunks,
then score each ordering against several definitions of "conjoint":

  nested_bpq1..5 : paper's design (nested quires over the contiguous text)
  blk2           : fixed adjacent pairs (== singulion)
  src_w{W}       : random pairs, leaves >=2 apart, within W chunks (inf: anywhere)
  blk{B}         : random pairs inside consecutive blocks of B leaves

Replicates are seeds (NMF init, SA init, pairing draws), NOT independent texts:
there is only one generated text, so spreads reflect algorithmic noise only.

Requires foliation.py, sweep.py, sweep_v2.py, scribe_control.py alongside.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd

import foliation as fol
import sweep as sw
import sweep_v2 as sv2
import scribe_control as sc

HERE = Path(__file__).parent


def load_trimmed(path: Path) -> list[str]:
    lines = [l for l in path.read_text(encoding="utf-8").splitlines()
             if not l.lstrip().startswith("#")]
    return fol.tokenize_clean("\n".join(lines))


def run_task(tokens, wpf, rep, cfg):
    t0 = time.time()
    n = len(tokens) // wpf
    chunks = [tokens[i * wpf:(i + 1) * wpf] for i in range(n)]
    rows = [(f, w) for f, t in enumerate(chunks, 1) for w in t]
    df = pd.DataFrame(rows, columns=["doc_id", "token"])
    counts_full = df.groupby(["doc_id", "token"]).size().unstack(fill_value=0).sort_index()
    counts = sw.prune_vocabulary(counts_full)
    doc_ids = counts.index.tolist()
    n_docs = len(doc_ids)
    k_max = min(sw.K_MAX, n_docs, counts.shape[1])
    k_min = min(sw.K_MIN, k_max)

    st = np.random.SeedSequence([cfg["seed"], rep, wpf]).generate_state(4)
    sw.rng = np.random.default_rng(int(st[2]))
    init_rng = np.random.default_rng(int(st[1]))
    dt, _e, wts = sw.build_nmf_ensemble(counts, k_min, k_max, sw.NMF_SEEDS,
                                        base_seed=int(st[0] % 2**31))
    svd_order = sw._svd_initial_order(dt, wts, n_docs, k_max)
    inits = [svd_order[:]] + [[int(x) for x in init_rng.permutation(n_docs)]
                              for _ in range(cfg["restarts"] - 1)]

    prng = np.random.default_rng(int(st[3]))
    src_of = ["T"] * n
    pairings = {f"nested_bpq{b}": [sc.nested_pairing(n, b)] for b in (1, 2, 3, 4, 5)}
    for W in sc.WINDOW_CHOICES:
        if W == 1:
            continue
        name = "src_w" + ("inf" if W is None else str(W))
        pairings[name] = [sc.source_pairing(src_of, W, prng)[0] for _ in range(sc.N_PAIRINGS)]
    for B in sc.BLOCKS:
        pairings[f"blk{B}"] = [sc.block_pairing(src_of, B, prng) for _ in range(sc.N_PAIRINGS)]

    out = []
    for metric in cfg["metrics"]:
        D = sv2.weighted_distance_matrix(dt, wts, metric, n_docs)
        res = []
        for r, init in enumerate(inits):
            order, cost = sw._sa_seriate(D, init, seed=int(st[3] % 2**31) + r,
                                         n_steps_multiplier=200)
            res.append((order, float(cost)))
        best_r = int(np.argmin([c for _, c in res]))
        for r, (order, cost) in enumerate(res):
            for name, lst in pairings.items():
                cs = []
                for p in lst:
                    adj = sw.bifolio_adjacency_rate(order, doc_ids, p)
                    b = sv2.exact_random_baseline(p)
                    cs.append(adj / b if b > 0 else np.nan)
                out.append(dict(wpf=wpf, rep=rep, metric=metric, restart=r,
                                best_cost_restart=(r == best_r), pairing=name,
                                carob=float(np.nanmean(cs)), n_folios=n_docs, cost=cost))
    return out, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", type=Path, default=HERE / "texts/self-citation/sc2_output_seed1404.txt")
    ap.add_argument("--wpf", type=int, nargs="+", default=[370, 374, 378, 382, 386])
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--restarts", type=int, default=8)
    ap.add_argument("--metrics", nargs="+", default=["hellinger", "js", "l1"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=HERE / "out_selfcite")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    toks = load_trimmed(a.text)
    print(f"tokens after trimming metadata: {len(toks)}", flush=True)
    cfg = dict(seed=a.seed, restarts=a.restarts, metrics=a.metrics)
    tasks = [(w, r) for w in a.wpf for r in range(a.reps)]
    t_all = time.time()
    for k, (w, r) in enumerate(tasks, 1):
        f = a.out / f"task_wpf{w}_rep{r}.csv"
        if f.exists():
            continue
        rows, secs = run_task(toks, w, r, cfg)
        pd.DataFrame(rows).to_csv(f, index=False)
        print(f"[{k}/{len(tasks)}] wpf={w} rep={r} {secs:.0f}s elapsed {(time.time()-t_all)/60:.1f}m", flush=True)
    pd.concat([pd.read_csv(p) for p in sorted(a.out.glob("task_*.csv"))],
              ignore_index=True).to_csv(a.out / "results_full.csv", index=False)
    print("done")


if __name__ == "__main__":
    main()
