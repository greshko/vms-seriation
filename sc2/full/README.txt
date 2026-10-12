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

SC2 - self-citation generator for Voynich-B-like text

Requirements: Python 3.9+, numpy, pandas, scipy, rapidfuzz  (pip install numpy pandas scipy rapidfuzz)

Keep all the .py files, best_params.json and targets.json together in one folder.
The scripts find each other from any working directory.

Generating text needs no CSV: the fitted tables and Voynich B's layout (words per line, sections, paragraph
starts; counts and structure only, no text) are bundled in sc2_model.json.
  python generate_sc2.py - 1 out.txt            (23,202 tokens)
  python generate_sc2.py - 1 out.txt 37500      (any length)
To refit from a transcription instead, pass the CSV as the first argument, or build a new model file:
  python build_model.py voynich_b_tokenized.csv my_model.json
  python generate_sc2.py my_model.json 1 out.txt
The diagnostics and tests compare against Voynich B, so they still need the CSV.

Where a CSV is needed, voynich_b_tokenized.csv is found, in order:
  1. a path given on the command line (generate_sc2.py, test_sc2.py)
  2. the VMS_CSV environment variable
  3. the current working directory
  4. this folder

Generate a text:
  python generate_sc2.py [- | model.json | transcription.csv] [seed] [out.txt] [n_tokens]
Run the checks:
  python test_sc2.py [csv]

Core files: sc2.py (generator), metrics.py (glyphs, loading, statistics), best_params.json (settings),
harvest.py (word parse used by coherence.py). The remaining scripts are diagnostics and tuning tools.

Paragraphs and sections: generate_sc2.py reads the page layout, assigns each page to its Voynich B section
(by folio number), and infers paragraph starts (a page's first line, or a line with a p/f gallows after a short
line). SC2 then cites partly from a section-wide pool and keeps page-local copying inside the current paragraph
(with a partial leak). Measure it with:  python selfsim_fast.py [csv] generated.txt
best_params_previous_v6.json holds the settings from before this change.

Vocabulary (familiarity and growth), current default:
  - graded familiarity: how readily a word is re-used depends on how often it has been written so far in the current
    section (weights calibrated to Voynich B's re-use curve); forms seen once are strongly disfavoured
  - a mild frequency memory: some long-memory draws favour the section's commonest words, usually written intact
  Expected vocabulary: about 5,000 word types at 23,202 tokens and about 7,200-7,500 at 37,500 tokens.
Length: python generate_sc2.py voynich_b_tokenized.csv 1 out.txt 37500
  (longer texts continue B's layout; the continued pages count as new sections, i.e. new material)
Check vocabulary:  python vocab_tools.py voynich_b_tokenized.csv out.txt
best_params_previous_v7.json holds the settings from before this change.

Pair copying (current default): the scribe remembers which word followed which (within lines) and, after writing a word,
sometimes writes a remembered successor of it directly, favouring pairs written often (p_pair, pair_gamma). This produces
Voynich-B-like word collocations (e.g. recurring two-word sequences) on top of the glyph-level boundary coupling.
best_params_previous_v8.json holds the settings from before this change.
