"""Theme sweep: each config on 2 seeds; appends JSON lines."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))   # find sibling modules from any working directory
import sys, json
from retune2 import *
base = load_params(); cfgs = json.loads(open(sys.argv[1]).read())
for name, kw in cfgs.items():
    for sd in (8001, 8002):
        kw2 = {**kw}; 
        if 'theme_sd' in kw2: kw2['theme_sd'] = tuple(kw2['theme_sd'])
        m = measure(S.SC2(T, S.Params(**{**base, **kw2, 'seed': sd})).generate(skel, SEED_LINE))
        with open(sys.argv[2], 'a') as f: f.write(json.dumps(dict(name=name, seed=sd, params=kw, metrics=m)) + '\n')
