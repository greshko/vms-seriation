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

"""Build an SC2 model file from a transcription CSV (columns: folio, line, word_line, word, token).

Usage: python build_model.py [csv | -] [out.json]      (default output: sc2_model.json next to this script)
The model holds the fitted tables (counts only) and the layout (tokens per line, sections, paragraph starts);
generate_sc2.py can then run without the CSV."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from metrics import load_layout, data_path
from sc2 import fit_tables, save_model
csv = _sys.argv[1] if len(_sys.argv) > 1 and _sys.argv[1] != "-" else None
out = _sys.argv[2] if len(_sys.argv) > 2 else _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "sc2_model.json")
pages, sections, starts = load_layout(csv)
save_model(out, fit_tables(pages), [[len(l) for l in p] for p in pages], sections, starts, source=_os.path.basename(data_path(csv)))
print(f"wrote {out}: {sum(map(sum, [[len(l) for l in p] for p in pages]))} tokens of layout on {len(pages)} pages, {_os.path.getsize(out) // 1024} KB")
