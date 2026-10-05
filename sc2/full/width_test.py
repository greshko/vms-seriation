import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))   # find sibling modules from any working directory
import json, random, numpy as np
from scipy.stats import spearmanr
from metrics import *
import sc2 as S
from line_profile import real
def width_stats(corp):
    """(1) corr(last-word length, glyphs in the rest of the line), controlling for token count;
       (2) spread of line width (glyphs) relative to a within-page shuffle of words across lines (same line sizes)."""
    rest, last, ntok, wid = [], [], [], []
    for p in corp:
        for l in p:
            if len(l) < 4: continue
            g = [len(glyphs(w)) for w in l]; rest.append(sum(g[:-1]) / (len(l) - 1)); last.append(g[-1]); ntok.append(len(l)); wid.append(sum(g) + len(l) - 1)
    rho = spearmanr(rest, last).correlation
    # width spread vs shuffle within page, per token count
    rng = random.Random(0); sd_real, sd_shuf = [], []
    for p in corp:
        lines = [l for l in p if len(l) >= 4]
        if len(lines) < 3: continue
        pool = [w for l in lines for w in l]; rng.shuffle(pool); it = iter(pool)
        sh = [[next(it) for _ in l] for l in lines]
        W = lambda L: [sum(len(glyphs(w)) for w in l) / len(l) for l in L]
        sd_real.append(np.std(W(lines))); sd_shuf.append(np.std(W(sh)))
    return rho, np.mean(sd_real) / np.mean(sd_shuf)
T = S.fit_tables(real); skel = [[len(l) for l in p] for p in real]; seed = "pchal shal shorchdy okeor okain shedy pchedy qotchedy qotar ol lkar".split()
best = load_params()
c = S.SC2(T, S.Params(**{**best, 'seed': 31, 'lam_len': 3, 'delta': 2.5})).generate(skel, seed)
rng = random.Random(1); shuf = []
for p in real:
    pool = [w for l in p for w in l]; rng.shuffle(pool); it = iter(pool); shuf.append([[next(it) for _ in l] for l in p])
for name, corp in [('Voynich B', real), ('SC2', c), ('B, words shuffled across lines of a page', shuf)]:
    rho, ratio = width_stats(corp)
    print(f"{name:42s} corr(mean length of other words, last-word length) = {rho:+.3f} | line-width spread vs within-page shuffle = {ratio:.3f}")
