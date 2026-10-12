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

"""Self-similarity of Voynich B and of a generated text poured into B's layout (pages, lines, inferred paragraphs, sections).

Usage: python selfsim_fast.py [csv | -] [generated.txt ...]
  generated texts are in generate_sc2.py's format (one line per manuscript line, blank line between pages)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import random, numpy as np
from collections import Counter, defaultdict
def flatten(pages, starts, sections):
    lines, page_id, para_id, sec = [], [], [], []; pid = -1
    for pi, (p, st) in enumerate(zip(pages, starts)):
        for li, l in enumerate(p):
            pid += bool(st[li]); lines.append(l); page_id.append(pi); para_id.append(pid); sec.append(sections[pi])
    return lines, page_id, para_id, sec
def exact_reuse(lines, unit):
    seen = defaultdict(set); ex = n = 0
    for li, ws in enumerate(lines):
        s = seen[unit[li]]
        for w in ws:
            if s: n += 1; ex += w in s
            s.add(w)
    return ex / n
def shuffled(lines, group, seed):
    rng = random.Random(seed); out = [list(ws) for ws in lines]; byg = defaultdict(list)
    for li, g in enumerate(group): byg[g].append(li)
    for lis in byg.values():
        pool = [w for li in lis for w in out[li]]; rng.shuffle(pool); it = iter(pool)
        for li in lis: out[li] = [next(it) for _ in out[li]]
    return out
def fast_selfsim(pages, starts, sections, reps=2):
    lines, page_id, para_id, sec = flatten(pages, starts, sections); line_id = list(range(len(lines)))
    r = lambda unit, grp: exact_reuse(lines, unit) / np.mean([exact_reuse(shuffled(lines, grp, s), unit) for s in range(reps)])
    out = {'line / page': r(line_id, page_id), 'paragraph / page': r(para_id, page_id), 'page / section': r(page_id, sec), 'paragraph / section': r(para_id, sec)}
    J = lambda a, b: len(set(a) & set(b)) / len(set(a) | set(b)); ov = defaultdict(list)
    for i in range(len(lines) - 1):
        if not lines[i] or not lines[i + 1]: continue
        k = 'overlap across page' if page_id[i] != page_id[i + 1] else ('overlap within para' if para_id[i] == para_id[i + 1] else 'overlap across para')
        ov[k].append(J(lines[i], lines[i + 1]))
    out.update({k: np.mean(v) for k, v in ov.items()})
    pg = defaultdict(Counter)
    for li, ws in enumerate(lines): pg[page_id[li]].update(ws)
    cos = lambda a, b: sum(a[w] * b[w] for w in a if w in b) / np.sqrt(sum(v * v for v in a.values()) * sum(v * v for v in b.values()))
    ids = sorted(pg); secp = {page_id[li]: sec[li] for li in range(len(lines))}
    out['page cosine (1-5 apart)'] = np.mean([cos(pg[a], pg[b]) for d in range(1, 6) for a, b in zip(ids, ids[d:]) if secp[a] == secp[b]])
    return out


def read_generated(path):
    return [[l.split() for l in pg.split("\n")] for pg in open(path).read().split("\n\n")]


if __name__ == "__main__":
    import sys, pandas as pd
    from metrics import load_layout
    csv = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "-" else None
    pages, sections, starts = load_layout(csv)
    rows = {"Voynich B": fast_selfsim(pages, starts, sections)}
    for f in sys.argv[2:]:
        c = read_generated(f)
        assert [[len(l) for l in p] for p in c] == [[len(l) for l in p] for p in pages], f"{f} does not follow the CSV's page/line layout"
        rows[f] = fast_selfsim(c, starts, sections)
    pd.set_option("display.width", 200); print(pd.DataFrame(rows).round(3).to_string())
