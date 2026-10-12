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
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein
from retune2 import *
P0 = S.Params(**load_params())
def tail(c):
    t = tokens(c); cnt = Counter(t); H = [w for w, k in cnt.items() if k == 1]; REC = [w for w, k in cnt.items() if k >= 2]; RS = set(REC)
    def strict(w):
        g = glyphs(w); return any(''.join(g[:k]) in RS and ''.join(g[k:]) in RS for k in range(3, len(g) - 2))
    def gate_fail(w):
        lp, ill = T['shape'].logp(w); return ill or lp < P0.shape_floor or len(glyphs(w)) > P0.len_max
    samp = H if len(H) <= 1500 else random.Random(0).sample(H, 1500)
    far = np.mean([process.extractOne(w, REC, scorer=Levenshtein.distance)[1] >= 2 for w in samp])
    init = Counter(l[0] for p in c for l in p if l); fin = Counter(l[-1] for p in c for l in p if len(l) >= 3)
    conc = [w for w in H if strict(w)]
    edge = np.mean([(init[w] + fin[w]) > 0 for w in conc]) if conc else np.nan
    hw = [w for w in t if 'h' in glyphs(w)]
    return dict(h_tokens=len(hw), h_initial=sum(glyphs(w)[0] == 'h' for w in hw), hapax_types=len(H), hapax_share_of_types=len(H) / len(cnt), strict_concat=np.mean([strict(w) for w in H]),
                gate_fail=np.mean([gate_fail(w) for w in H]), far_2plus=far, concat_at_line_edge=edge)
if __name__ == '__main__':
    if sys.argv[1] == 'ref':
        r = tail(real); r.update(measure(real)); json.dump(r, open('tail_ref.json', 'w'), indent=1); print({k: round(v, 3) for k, v in r.items() if k in ('hapax_share_of_types', 'strict_concat', 'gate_fail', 'far_2plus', 'concat_at_line_edge')})
    else:
        base = load_params()
        cfgs = json.load(open(sys.argv[2]))
        for name, kw in cfgs.items():
            for sd in ((8101, 8102) if len(sys.argv) < 5 else tuple(int(x) for x in sys.argv[4].split(','))):
                g = S.SC2(T, S.Params(**{**base, **kw, 'seed': sd})); c = g.generate(skel, SEED_LINE)
                m = measure(c); m.update(tail(c)); m['routes'] = {k: v for k, v in g.stats.items() if k.startswith('oneoff')}
                with open(sys.argv[3], 'a') as f: f.write(json.dumps(dict(name=name, seed=sd, params=kw, metrics=m)) + '\n')
