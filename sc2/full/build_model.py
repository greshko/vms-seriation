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
