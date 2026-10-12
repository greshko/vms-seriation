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
import json, sys
from retune2 import *
base = load_params(); tgt = json.load(open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'targets.json')))
d = sorted([json.loads(l) for l in open('retune2.jsonl')], key=lambda x: x['loss'])[:5]
cands = [('current settings', {})] + [(f"run {x['run']}", x['params']) for x in d]
for name, p in cands:
    for sd in (7001, 7002, 7003):
        m = measure(S.SC2(T, S.Params(**{**base, **p, 'seed': sd})).generate(skel, SEED_LINE))
        with open('validate2.jsonl', 'a') as f: f.write(json.dumps(dict(name=name, seed=sd, loss=loss(m, tgt), params=p, metrics=m)) + '\n')
