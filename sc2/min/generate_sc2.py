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
