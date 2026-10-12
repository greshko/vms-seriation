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
import json, random, math, sys
from collections import Counter
from metrics import *
import sc2 as S
real = load_real(sys.argv[1] if len(sys.argv) > 1 else None); T = S.fit_tables(real)
skel = [[len(l) for l in p] for p in real]; seed = "pchal shal shorchdy okeor okain shedy pchedy qotchedy qotar ol lkar".split()
P = S.Params(**load_params(seed=7))
g = S.SC2(T, P)
# instrument source choice
kinds = Counter(); orig = g._choose_source   # the raw draw, before familiarity/one-off redraws
def wrapped(page, line_no, pos, prev_src, history):
    t, loc, how = orig(page, line_no, pos, prev_src, history)
    above = line_no > 0 and pos < len(page[line_no - 1])
    kinds[(above, how)] += 1; return t, loc, how
g._choose_source = wrapped
corp = g.generate(skel, seed); toks = tokens(corp)
print('TEST 1  global-draw rate (excluding span and line-anchor draws, which come first), target p_global =', round(P.p_global, 3))
for above in (True, False):
    tot = sum(v for (a, h), v in kinds.items() if a == above and h not in ('span', 'anchor')); gl = kinds[(above, 'global')] + kinds[(above, 'freq')]   # frequency-memory draws are global draws
    print(f"   same-position source {'available' if above else 'missing  '}: global {gl/tot:.3f} of {tot} draws")
print('TEST 2  look-back from a 100-token history, scale 2000 (share landing on the oldest token; uniform = 0.010)')
h = list(range(100)); P2 = S.Params(global_scale=2000, seed=1); g2 = S.SC2(T, P2)
new = Counter(g2._global(h) for _ in range(20000))
old = Counter(h[-min(len(h), int(random.expovariate(1 / 2000)) + 1)] for _ in range(20000))
print(f"   old rule {old[0]/20000:.3f}   fixed rule {new[0]/20000:.3f}   (fixed, max share of any index {max(new.values())/20000:.3f})")
hcount = lambda ts: (sum('h' in glyphs(w) for w in ts), sum(glyphs(w)[0] == 'h' for w in ts))
print('TEST 3  standalone-h slips (tokens with one, of which word-initial): SC2', hcount(toks), '| Voynich B', hcount(tokens(real)))
print('        benches are never split: every h-bearing SC2 token re-parses with ch/sh/cth/ckh/cph/cfh intact:',
      all(''.join(glyphs(w)) == w for w in toks))
init = [p[0][0] for p in corp if p and p[0]]
print(f"   page-initial words opening with a gallows glyph: {sum(glyphs(w)[0] in GALLOWS for w in init)}/{len(init)}; examples: {' '.join(init[:10])}")
print('TEST 4  fallbacks and slips:', dict(g.stats), '| tokens failing the gate (h slips and one-off compounds/bypass forms excepted):', sum(not g.legal(w) for w in toks if 'h' not in glyphs(w) and w not in g.oneoff), 'of', len(toks))
