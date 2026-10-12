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

"""Full joint re-tune of SC2 with global and line-level targets. Appends one JSON line per run."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))   # find sibling modules from any working directory
import sys, json, random, math, time, numpy as np
from collections import Counter
from metrics import *
import sc2 as S
from line_profile import profile, real
from coherence import coherence
T = S.fit_tables(real); skel = [[len(l) for l in p] for p in real]; SEED_LINE = "pchal shal shorchdy okeor okain shedy pchedy qotchedy qotar ol lkar".split()
VOC = set(tokens(real))
def measure(c):
    s = summarize(c, VOC, curve=False); p = profile(c); co = {k: (v if v == v else 1.0) for k, v in coherence(c, reps=3).items()}
    return dict(MI_boundary=s['MI_boundary_excess'], ttr=s['ttr'], hapax=s['hapax_frac'], burstiness=s['burstiness'], p_ed1=s['p_ed1'],
                len_mean=s['len_mean'], alpha=long_alpha(c), jsd_init=s['jsd_line_initial'], jsd_final=s['jsd_line_final'],
                len_1st=p.loc['1st', 'length'], len_2nd=p.loc['2nd', 'length'], len_last=p.loc['last', 'length'], jsd_first_2nd=p.loc['2nd', 'jsd_first'],
                m_last=p.loc['last', 'm_last'], coh_length=co['word length'], coh_family=co['qok/ot family'], coh_y=co['ends -y'], coh_chsh=co['opens ch/sh'],
                zipf=s['zipf'], H2=s['H2'], H3=s['H3'], p_repeat=s['p_repeat'], types=s['types'])
W = dict(MI_boundary=1, ttr=1, hapax=1, burstiness=1, p_ed1=1, len_mean=2, alpha=1, jsd_init=1, jsd_final=1,
         len_1st=3, len_2nd=3, len_last=3, jsd_first_2nd=0.5, m_last=1)
def loss(m, t):
    L = sum(w * abs(m[k] - t[k]) / abs(t[k]) for k, w in W.items())
    L += sum(0.5 * abs((m[k] - 1) - (t[k] - 1)) / (t[k] - 1) for k in ('coh_length', 'coh_family', 'coh_y'))
    L += 0.5 * max(0.0, m['coh_chsh'] - 1.0) / 0.1          # B's ch/sh openings are not line-coherent
    return L
def sample(r):
    return dict(K=r.choice([8, 12, 16]), tau=math.exp(r.uniform(math.log(4), math.log(40))), p_same_pos=r.uniform(.05, .3), p_span=r.uniform(0, .15),
                p_global=r.uniform(.1, .45), global_scale=math.exp(r.uniform(math.log(1000), math.log(16000))), beta=r.uniform(2, 5), gamma=r.uniform(1.5, 4),
                delta=r.uniform(1, 3.5), lam_len=r.uniform(1, 4), p_anchor=r.uniform(0, .45), tau_anchor=r.uniform(1, 8), p_extra_mut=r.uniform(.1, .45),
                p_cross_class=r.uniform(.2, .7), p_flow=r.uniform(.2, .6), p_flow_pos=r.uniform(.4, .9), shape_floor=r.uniform(-2.6, -2.2))
def perturb(d, r, scale=0.15):
    lim = dict(K=(8, 16), tau=(3, 60), p_same_pos=(0, .4), p_span=(0, .25), p_global=(.05, .6), global_scale=(500, 30000), beta=(1, 6), gamma=(1, 5),
               delta=(.5, 4), lam_len=(.5, 5), p_anchor=(0, .6), tau_anchor=(.5, 10), p_extra_mut=(.05, .6), p_cross_class=(.1, .8), p_flow=(.1, .8),
               p_flow_pos=(.2, 1), shape_floor=(-2.8, -2.0))
    out = dict(d)
    for k in r.sample(list(lim), 4):
        lo, hi = lim[k]
        if k == 'K': out[k] = r.choice([8, 12, 16]); continue
        out[k] = float(min(hi, max(lo, d[k] + r.gauss(0, scale * (hi - lo)))))
    return out
if __name__ == '__main__':
    mode, start, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    base = load_params(); fixed = {k: base[k] for k in ('op_w', 'h_scale', 'p_slip_copy', 'p_para_gallows', 'len_max', 'max_rounds') if k in base}
    tgt = json.load(open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'targets.json')))
    for i in range(start, start + n):
        r = random.Random(i)
        if mode == 'random': d = sample(r)
        else:
            done = [json.loads(l) for l in open('retune2.jsonl')]; top = sorted(done, key=lambda x: x['loss'])[:5]
            d = perturb(r.choice(top)['params'], r)
        t0 = time.time()
        m = measure(S.SC2(T, S.Params(**{**base, **fixed, **d, 'seed': i})).generate(skel, SEED_LINE))
        with open('retune2.jsonl', 'a') as f: f.write(json.dumps(dict(run=i, mode=mode, loss=loss(m, tgt), params=d, metrics=m, secs=time.time() - t0)) + '\n')
