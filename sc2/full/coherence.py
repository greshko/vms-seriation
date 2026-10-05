import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))   # find sibling modules from any working directory
import json, random, numpy as np
from metrics import *
from harvest import parse
FEAT = {'word length': lambda w: len(glyphs(w)),
        'qok/ot family': lambda w: parse(w)[0] in ('qok', 'qot', 'ok', 'ot') and parse(w)[1] in ('e', 'ee', 'eee'),
        'has gallows': lambda w: any(g in GALLOWS for g in glyphs(w)),
        'opens ch/sh': lambda w: glyphs(w)[0] in ('ch', 'sh'),
        'ends -y': lambda w: w.endswith('y'),
        '-aiin/-ain': lambda w: w.endswith('ain') or w.endswith('aiin'),
        'ends -l/-r': lambda w: w[-1] in 'lr'}
def coherence(corp, reps=5):
    """Between-line spread of each feature's line mean, relative to shuffling words among the lines of the same page
    (excluding each line's first and last word, so line-edge effects don't count)."""
    out = {}
    rng = random.Random(0)
    for name, f in FEAT.items():
        real_sd, shuf_sd = [], []
        for p in corp:
            lines = [[float(f(w)) for w in l[1:-1]] for l in p if len(l) >= 5]
            if len(lines) < 3: continue
            real_sd.append(np.std([np.mean(l) for l in lines]))
            pool = [x for l in lines for x in l]; s = []
            for _ in range(reps):
                rng.shuffle(pool); it = iter(pool); s.append(np.std([np.mean([next(it) for _ in l]) for l in lines]))
            shuf_sd.append(np.mean(s))
        out[name] = np.mean(real_sd) / np.mean(shuf_sd)
    return out
if __name__ == '__main__':
    import sc2 as S, pandas as pd
    from line_profile import real
    T = S.fit_tables(real); skel = [[len(l) for l in p] for p in real]; seed = "pchal shal shorchdy okeor okain shedy pchedy qotchedy qotar ol lkar".split()
    c = S.SC2(T, S.Params(**{**load_params(), 'seed': 31, 'lam_len': 3, 'delta': 2.5})).generate(skel, seed)
    df = pd.DataFrame({'Voynich B': coherence(real), 'SC2': coherence(c)})
    print('line coherence (between-line spread / within-page shuffle; 1 = none), line edges excluded'); print(df.round(3).to_string())
