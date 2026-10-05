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
