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
import json, sys, numpy as np
from collections import Counter
from rapidfuzz.distance import Levenshtein
from tail_measure import *
def lag2(c):
    import random as _r
    pr = [(last_g(l[i]), first_g(l[i + 2])) for p in c for l in p for i in range(len(l) - 2)]
    a = [x for x, _ in pr]; b = [y for _, y in pr]; rg = _r.Random(0)
    return mi_from_pairs(pr) - np.mean([mi_from_pairs(list(zip(a, rg.sample(b, len(b))))) for _ in range(3)])
def rep(c):
    cnt = Counter(tokens(c)); N = sum(cnt.values())
    pairs = [(l[i], l[i + 1], i, len(l)) for p in c for l in p for i in range(len(l) - 1)]
    band = lambda k: '2-9' if k < 10 else '10-99' if k < 100 else '100+'
    tot, rp, pt, pr = Counter(), Counter(), Counter(), Counter()
    for a, b, i, n in pairs:
        if cnt[a] > 1: tot[band(cnt[a])] += 1; rp[band(cnt[a])] += a == b
        k = 'first' if i == 0 else 'last' if i == n - 2 else 'medial'; pt[k] += 1; pr[k] += a == b
    return dict(rep_all=np.mean([a == b for a, b, _, _ in pairs]), ed1=np.mean([Levenshtein.distance(a, b) == 1 for a, b, _, _ in pairs]),
                rep_chance=sum((k / N) ** 2 for k in cnt.values()), **{f'rep_freq_{k}': rp[k] / max(1, tot[k]) for k in ('2-9', '10-99', '100+')},
                **{f'rep_pos_{k}': pr[k] / max(1, pt[k]) for k in ('first', 'medial', 'last')},
                lag2=lag2(c), triples=sum(1 for p in c for l in p for i in range(len(l) - 2) if l[i] == l[i + 1] == l[i + 2]))
if __name__ == '__main__':
    if sys.argv[1] == 'ref':
        r = rep(real); json.dump(r, open('rep_ref.json', 'w'), indent=1); print({k: round(v, 4) for k, v in r.items()})
    else:
        base = load_params()
        for name, kw in json.load(open(sys.argv[2])).items():
            for sd in tuple(int(x) for x in sys.argv[4].split(',')):
                g = S.SC2(T, S.Params(**{**base, **kw, 'seed': sd})); c = g.generate(skel, SEED_LINE)
                m = measure(c); m.update(tail(c)); m.update(rep(c)); m['n_dup'] = g.stats['reduplicated']
                with open(sys.argv[3], 'a') as f: f.write(json.dumps(dict(name=name, seed=sd, params=kw, metrics=m)) + '\n')
