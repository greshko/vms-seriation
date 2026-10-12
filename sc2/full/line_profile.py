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

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))   # find sibling modules from any working directory
import json, sys, numpy as np, pandas as pd
from collections import Counter
from metrics import *
import sc2 as S
real = load_real()
POS = ['1st', '2nd', '3rd', 'mid', '3rd-last', '2nd-last', 'last']
def pos_of(i, n):
    if n < 7: return None
    return {0: '1st', 1: '2nd', 2: '3rd', n - 3: '3rd-last', n - 2: '2nd-last', n - 1: 'last'}.get(i, 'mid')
def profile(corp):
    L, F, E, R, S1 = {p: [] for p in POS}, {p: Counter() for p in POS}, {p: Counter() for p in POS}, {p: [] for p in POS}, {p: [] for p in POS}
    for pg in corp:
        for l in pg:
            for i, w in enumerate(l):
                p = pos_of(i, len(l))
                if p is None: continue
                L[p].append(len(glyphs(w))); F[p][first_g(w)] += 1; E[p][last_g(w)] += 1
                R[p].append(i > 0 and w == l[i - 1])
    mid_f = F['mid']; mid_e = E['mid']
    rows = {}
    for p in POS:
        n = sum(F[p].values())
        rows[p] = dict(length=np.mean(L[p]), jsd_first=jsd(F[p], mid_f), jsd_last=jsd(E[p], mid_e),
                       gallows_first=sum(F[p][g] for g in GALLOWS) / n, sdy_first=sum(F[p][g] for g in 'sdy') / n,
                       qo_first=F[p]['q'] / n, ch_sh_first=(F[p]['ch'] + F[p]['sh']) / n, m_last=E[p]['m'] / n, n_last=E[p]['n'] / n, y_last=E[p]['y'] / n)
    return pd.DataFrame(rows).T
if __name__ == '__main__':
    t = tokens(real); hw = [w for w in t if 'h' in glyphs(w)]; hc = Counter(hw)
    print(f"B standalone-h words: {len(hw)} tokens, {len(hc)} types, {sum(c for c in hc.values() if c > 1)} tokens in repeated types, max repeat {max(hc.values())}")
    tpl = [len(l) for p in real for l in p]
    print(f"B tokens per line: mean {np.mean(tpl):.1f}; lines with >=7 tokens (profiled): {np.mean(np.array(tpl) >= 7):.0%}")
    T = S.fit_tables(real); skel = [[len(l) for l in p] for p in real]; seed = "pchal shal shorchdy okeor okain shedy pchedy qotchedy qotar ol lkar".split()
    c = S.SC2(T, S.Params(**{**load_params(), 'seed': 31})).generate(skel, seed)
    pb, ps = profile(real), profile(c); pickle_ok = True
    pd.set_option('display.width', 220)
    print('\nVoynich B'); print(pb.round(3).to_string()); print('\nSC2'); print(ps.round(3).to_string())
