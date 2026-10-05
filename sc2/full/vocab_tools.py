"""Vocabulary diagnostics: the re-use (attachment) curve and type growth, for Voynich B and generated texts.

Usage: python vocab_tools.py [csv | -] [generated.txt ...]
  generated texts in generate_sc2.py's format (one line per manuscript line, blank line between pages), any length.

Re-use curve: for words already seen k times, the per-word rate of being used again divided by k, normalised so that
the 6-10 band = 1 (flat = re-use proportional to past frequency). Growth: number of word types after n tokens."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import numpy as np, pandas as pd
from collections import Counter
BINS = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, 10 ** 9)]
CHECKPOINTS = [5000, 10000, 15000, 20000, 23202, 30000, 37500]


def reuse_curve(toks):
    cnt, nk, hits, expo = Counter(), Counter(), Counter(), Counter()
    for w in toks:
        k = cnt[w]
        for kk, n in nk.items(): expo[kk] += n
        if k: hits[k] += 1; nk[k] -= 1
        cnt[w] += 1; nk[k + 1] += 1
        if nk[k] == 0: nk.pop(k, None)
    out = {}
    for lo, hi in BINS:
        ks = [k for k in expo if lo <= k <= hi]
        e = sum(expo[k] for k in ks)
        kbar = sum(expo[k] * k for k in ks) / max(1, e)
        out[str(lo) if lo == hi else (f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+")] = (sum(hits[k] for k in ks) / e) / kbar if e else np.nan
    s = pd.Series(out); return s / s.loc["6-10"]


def growth(toks, checkpoints=CHECKPOINTS):
    seen, out, i = set(), {}, 0
    for n in checkpoints:
        if n > len(toks): break
        while i < n: seen.add(toks[i]); i += 1
        out[n] = len(seen)
    return out


def read_generated(path):
    return [w for pg in open(path).read().split("\n\n") for l in pg.split("\n") for w in l.split()]


if __name__ == "__main__":
    from metrics import load_real
    csv = _sys.argv[1] if len(_sys.argv) > 1 and _sys.argv[1] != "-" else None
    texts = {"Voynich B": [w for p in load_real(csv) for l in p for w in l]}
    for f in _sys.argv[2:]: texts[f] = read_generated(f)
    pd.set_option("display.width", 200)
    print("Re-use curve (flat = proportional; normalised to the 6-10 band)")
    curves = pd.DataFrame({k: reuse_curve(v) for k, v in texts.items()}).T
    if len(texts) > 1: curves["mean abs. diff. from B"] = (curves - curves.loc["Voynich B"]).abs().mean(axis=1)
    print(curves.round(3).to_string())
    print("\nWord types after n tokens")
    print(pd.DataFrame({k: growth(v) for k, v in texts.items()}).T.to_string())
