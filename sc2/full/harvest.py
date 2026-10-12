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
import re
from collections import Counter
import pandas as pd
from metrics import data_path
B = pd.read_csv(data_path()).word.astype(str).tolist()
# onset: optional initial glyph (s only if not the bench 'sh') + optional plain gallows
# middle: optional bench or bench-gallows + e-run (at least one of the two)
# ending: remainder
RX = re.compile(r'^(?P<pre>(?:qo|q|o|y|d|s(?!h)|l|r)?(?:k|t|p|f)?)(?P<mid>(?:ckh|cth|cph|cfh|ch|sh)?e{0,3})(?P<suf>.*)$')
def parse(w):
    m = RX.match(w); return m['pre'], m['mid'], m['suf']
def shape(w):
    p, m, s = parse(w)
    if p and m and s: return 'T'
    if (bool(p) != bool(m)) and s: return 'B'
    return 'U'
def role_counts():
    C = {r: Counter() for r in ['U', 'BP', 'BS', 'TO', 'TM', 'TS', 'BPx']}
    for w in B:
        p, m, s = parse(w); sh = shape(w)
        C['U'][w] += 1
        if sh == 'T':
            C['TO'][p] += 1; C['TM'][m] += 1; C['TS'][s] += 1
            C['BPx'][p + m] += 1; C['BS'][s] += 1          # bigram-only table variant: onset+middle as one prefix
        elif sh == 'B':
            C['BP'][p or m] += 1; C['BPx'][p or m] += 1; C['BS'][s] += 1
    return C
if __name__ == '__main__':
    sh = Counter(shape(w) for w in B); n = len(B)
    print('B token shapes:', {k: f'{v/n:.1%}' for k, v in sh.items()})
    C = role_counts()
    for r, c in C.items():
        if r == 'U': continue
        print(r, len(c), 'distinct, >=3:', sum(v >= 3 for v in c.values()), c.most_common(12))
