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

"""Generate a Voynich-B-like text with SC2.

Usage: python generate_sc2.py [source] [seed] [out.txt] [n_tokens]
  source   "-" or omitted: the bundled model sc2_model.json (no CSV needed)
           a .json file:   another model built with build_model.py
           a .csv file:    a transcription; tables are refitted and the layout read from it
  seed     random seed (default 0)
  out.txt  output file (default sc2_output.txt): one line per manuscript line, blank line between pages
  n_tokens length (default: the layout's own, 23,202 for Voynich B); longer texts continue the layout with its pages
           again, treated as new sections (new material); shorter texts truncate it."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))   # find sibling modules from any working directory
import sys
from metrics import load_layout, extend_skeleton, load_params
from sc2 import SC2, Params, fit_tables, load_model

HERE = _os.path.dirname(_os.path.abspath(__file__))
src = sys.argv[1] if len(sys.argv) > 1 else "-"
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
out = sys.argv[3] if len(sys.argv) > 3 else "sc2_output.txt"
n_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else None

if src == "-" or src.lower().endswith(".json"):
    model = _os.path.join(HERE, "sc2_model.json") if src == "-" else src
    tables, skeleton, sections, para_starts = load_model(model)
    origin = f"model {_os.path.basename(model)}"
else:
    pages, sections, para_starts = load_layout(src)                   # pages -> lines -> tokens; section per page; paragraph starts
    tables = fit_tables(pages); skeleton = [[len(l) for l in p] for p in pages]
    origin = f"CSV {_os.path.basename(src)}"
if n_tokens: skeleton, sections, para_starts = extend_skeleton(skeleton, sections, para_starts, n_tokens)

seed_line = "pchal shal shorchdy okeor okain shedy pchedy qotchedy qotar ol lkar".split()   # Timm & Schinner's f103v.P.9
corpus = SC2(tables, Params(**load_params(seed=seed))).generate(skeleton, seed_line, sections, para_starts)
with open(out, "w") as f:
    f.write("\n\n".join("\n".join(" ".join(line) for line in page) for page in corpus))
print(f"wrote {sum(len(l) for p in corpus for l in p)} tokens on {len(corpus)} pages to {out} (from {origin})")
