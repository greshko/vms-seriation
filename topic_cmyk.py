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

"""
topic_cmyk.py

Chunk a text (or a CSV of tokens grouped by an ID column) into chunks, fit
a 4-topic NMF (Non-negative Matrix Factorization) model across the chunks,
then map each chunk's topic-mixture distribution directly onto a CMYK
color (one axis per topic) so the material's topical progression can be
viewed as a strip of color.

Usage (plain text, N-word chunks):
    python topic_cmyk.py input.txt --chunk-size 200 --out progression.png

Usage (CSV, chunked by an ID column instead of word count):
    python topic_cmyk.py tokens.csv --group-col doc_id --token-col token \\
        --min-df 2 --max-df 0.8 --out progression.png

If no input is given, a small built-in demo text is used instead.
"""

import argparse
import re
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

N_TOPICS = 4  # fixed: one topic per CMYK channel

# A compact list of high-frequency Latin function words (pronouns,
# conjunctions, prepositions, common verb forms) — scikit-learn ships no
# built-in Latin stopword list, and TF-IDF alone doesn't fully suppress
# words this common in an inflected language.
LATIN_STOPWORDS = [
    "et", "in", "ad", "de", "ex", "cum", "sed", "vel", "vel", "ut", "ne",
    "si", "sic", "non", "nec", "neque", "quam", "quod", "quia", "quae",
    "qui", "quibus", "quo", "quos", "quas", "quorum", "quarum", "cui",
    "cuius", "hoc", "hic", "haec", "hunc", "hanc", "his", "huius", "ille",
    "illa", "illud", "illi", "illis", "ipse", "ipsa", "ipsum", "is", "ea",
    "id", "eius", "eum", "eam", "eos", "eas", "eorum", "earum", "ei",
    "sui", "sibi", "se", "suus", "sua", "suum", "suis", "atque", "ac",
    "aut", "autem", "enim", "etiam", "erat", "erant", "esse", "est",
    "sunt", "sit", "sint", "esset", "fuit", "fuerunt", "fuerat", "fieri",
    "factus", "facta", "factum", "per", "pro", "prae", "ab", "a", "post",
    "ante", "inter", "sub", "super", "apud", "usque", "tam", "tamen",
    "tantum", "ita", "iam", "quoque", "quidem", "igitur", "ergo", "nam",
    "nunc", "tunc", "ubi", "unde", "cum", "dum", "cur", "quomodo",
    "quidam", "quaedam", "quoddam", "omnis", "omnes", "omnia", "omnium",
    "multi", "multa", "multis", "alius", "alia", "aliud", "alii",
    "aliquis", "aliquid", "nullus", "nulla", "nullum", "unus", "una",
    "unum", "duo", "duae", "tres", "primus", "prima", "primum", "inquit",
    "dixit", "dicit", "dicens", "dicunt", "possum", "potest", "poterat",
    "potuit", "debet", "debuit", "vero", "quibusdam", "cunctis", "cuncta",
    "quisque", "quaeque", "quodque", "eodem", "eadem", "isdem", "idem",
]

# A compact list of high-frequency German function words (articles,
# pronouns, conjunctions, prepositions, auxiliary verb forms). Also covers
# common archaic/OCR-variant spellings (e.g. "und", "auch", "welcher").
GERMAN_STOPWORDS = [
    "der", "die", "das", "den", "dem", "des", "ein", "eine", "einer",
    "eines", "einem", "einen", "und", "oder", "aber", "auch", "als",
    "wie", "wenn", "dann", "so", "nicht", "kein", "keine", "keiner",
    "keinem", "keinen", "nur", "noch", "schon", "sehr", "mehr", "viel",
    "viele", "vielen", "alle", "aller", "allem", "allen", "alles",
    "man", "ich", "du", "er", "sie", "es", "wir", "ihr", "mich", "dich",
    "sich", "uns", "euch", "mein", "meine", "meiner", "meinem", "meinen",
    "dein", "deine", "sein", "seine", "seiner", "seinem", "seinen",
    "ihre", "ihrer", "ihrem", "ihren", "unser", "unsere", "euer", "eure",
    "in", "im", "an", "am", "auf", "aus", "bei", "bis", "durch", "für",
    "gegen", "mit", "nach", "seit", "von", "vom", "vor", "zu", "zum",
    "zur", "über", "unter", "zwischen", "ohne", "um", "ist", "sind",
    "war", "waren", "wird", "werden", "wurde", "wurden", "hat", "haben",
    "hatte", "hatten", "kann", "können", "konnte", "konnten", "muss",
    "müssen", "musste", "mussten", "soll", "sollen", "sollte", "sollten",
    "darf", "dürfen", "will", "wollen", "wollte", "wollten", "sei",
    "wäre", "wird", "sein", "worden", "diese", "dieser", "dieses",
    "diesem", "diesen", "jene", "jener", "jenes", "welche", "welcher",
    "welches", "welchem", "welchen", "was", "wer", "wen", "wem", "wessen",
    "wo", "wohin", "woher", "warum", "weil", "damit", "dass", "ob",
    "doch", "denn", "also", "zwar", "sondern", "sowie", "sowohl",
    "jedoch", "hier", "da", "dort", "jetzt", "immer", "wieder", "etwas",
    "nichts", "jemand", "niemand", "andere", "anderen", "anderer",
    "anderes", "selbst", "eigen", "eigenen", "eigener", "eigenes",
]


# --------------------------------------------------------------------------
# 1. Chunking
# --------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int) -> list[str]:
    """Split text into chunks of `chunk_size` words each (last chunk may be
    shorter). Whitespace is normalized; punctuation is kept attached to words
    so downstream tokenization in CountVectorizer can still work normally."""
    words = text.split()
    if not words:
        raise ValueError("Input text has no words to chunk.")
    chunks = [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]
    return chunks


def chunk_by_doc_id(csv_path: str, group_col: str = "doc_id", token_col: str = "token",
                     label_col: str | None = None) -> tuple[list[str], list[str], list]:
    """Read a CSV of one-token-per-row data and group rows into chunks by
    `group_col` (e.g. a folio/document id), concatenating `token_col` values
    within each group (in row order) into a single space-joined chunk.

    Groups are sorted ascending by `group_col`, which is assumed to reflect
    the material's natural reading order (e.g. increasing folio numbers).

    Returns (chunks, labels, group_ids):
      - labels are human-readable identifiers for each chunk (see below)
      - group_ids are the raw group_col values, e.g. for re-keying/re-ordering
        chunks by some other permutation later
    Label priority:
      - the value of `label_col` (first row in the group), if given
      - a "f1+f8"-style tag listing every distinct folio number found in a
        `full_folio` column, if present and label_col was not given
      - the raw group id, as a fallback
    """
    df = pd.read_csv(csv_path)
    if group_col not in df.columns:
        raise ValueError(f"Column '{group_col}' not found. Available columns: {list(df.columns)}")
    if token_col not in df.columns:
        raise ValueError(f"Column '{token_col}' not found. Available columns: {list(df.columns)}")

    chunks, labels, group_ids = [], [], []
    for gid, sub in df.groupby(group_col, sort=True):
        chunks.append(" ".join(sub[token_col].astype(str)))
        group_ids.append(gid)
        if label_col and label_col in df.columns:
            labels.append(str(sub[label_col].iloc[0]))
        elif "full_folio" in df.columns:
            nums = sorted({
                int(m.group(1)) for v in sub["full_folio"]
                if (m := re.match(r"f(\d+)", str(v)))
            })
            labels.append("+".join(f"f{n}" for n in nums) if nums else str(gid))
        else:
            labels.append(str(gid))
    return chunks, labels, group_ids


# --------------------------------------------------------------------------
# 2. Topic modeling
# --------------------------------------------------------------------------
def fit_topics(chunks: list[str], n_topics: int = N_TOPICS, random_state: int = 42,
               stopwords: str = "english", min_df=1, max_df=0.95,
               token_pattern: str = r"(?u)\b[a-zA-ZäöüÄÖÜß]{2,}\b", lowercase: bool = True):
    """Fit an NMF model over the chunks and return:
       - doc_topic: (n_chunks, n_topics) normalized topic-mixture matrix
       - nmf: the fitted NMF model
       - vectorizer: the fitted TfidfVectorizer

    min_df / max_df are passed straight through to TfidfVectorizer: min_df
    prunes terms appearing in fewer than that many chunks (e.g. min_df=2
    drops every term that appears in only a single chunk), and max_df prunes
    terms appearing in more than that fraction of chunks (e.g. max_df=0.8
    drops anything present in over 80% of chunks). This is a stopword-free,
    corpus-driven way to strip both one-off noise and ubiquitous "function"
    tokens — useful when there's no curated stopword list for the language
    or symbol system in question.
    """
    if len(chunks) < n_topics:
        print(
            f"Warning: only {len(chunks)} chunk(s) available but {n_topics} "
            f"topics requested. Results will be degenerate; consider a "
            f"smaller --chunk-size.",
            file=sys.stderr,
        )

    if stopwords == "english":
        stop_list = "english"
    elif stopwords == "latin":
        stop_list = LATIN_STOPWORDS
    elif stopwords == "german":
        stop_list = GERMAN_STOPWORDS
    else:
        stop_list = None

    vectorizer = TfidfVectorizer(
        stop_words=stop_list,
        token_pattern=token_pattern,
        max_df=max_df,
        min_df=min_df,
        lowercase=lowercase,
    )
    doc_term = vectorizer.fit_transform(chunks)

    if doc_term.shape[1] == 0:
        raise ValueError(
            "Vocabulary is empty after preprocessing — text may be too short, "
            "too repetitive/stopword-heavy, or min_df/max_df pruned everything."
        )

    nmf = NMF(
        n_components=n_topics,
        random_state=random_state,
        init="nndsvda",
        max_iter=500,
    )
    doc_topic = nmf.fit_transform(doc_term)

    # Unlike LDA, NMF's output is just non-negative — it has no built-in
    # constraint that each row sums to 1. We explicitly normalize each
    # chunk's topic-weight row into a proper distribution (so it can be
    # treated as CMYK proportions). Guard against an all-zero row (e.g. a
    # chunk with no vocabulary overlap with the fitted topics).
    row_sums = doc_topic.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    doc_topic = doc_topic / row_sums

    return doc_topic, nmf, vectorizer


def top_words_per_topic(nmf, vectorizer, n_words: int = 8) -> list[list[str]]:
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic in nmf.components_:
        top_idx = topic.argsort()[::-1][:n_words]
        topics.append([feature_names[i] for i in top_idx])
    return topics


# --------------------------------------------------------------------------
# 3. Topic mixture -> CMYK -> RGB
# --------------------------------------------------------------------------
def topic_mixture_to_cmyk(topic_dist: np.ndarray) -> np.ndarray:
    """A topic-mixture row already is a 4-vector that sums to 1 with all
    entries in [0, 1] — exactly the shape a CMYK tuple needs. We use it
    directly as (C, M, Y, K)."""
    return np.clip(topic_dist, 0.0, 1.0)


def cmyk_to_rgb(cmyk: np.ndarray) -> np.ndarray:
    """Standard CMYK -> RGB conversion. cmyk: (..., 4) array in [0,1].
    Returns RGB array in [0,1]."""
    c, m, y, k = cmyk[..., 0], cmyk[..., 1], cmyk[..., 2], cmyk[..., 3]
    r = (1 - c) * (1 - k)
    g = (1 - m) * (1 - k)
    b = (1 - y) * (1 - k)
    return np.stack([r, g, b], axis=-1)


# --------------------------------------------------------------------------
# 4. Visualization
# --------------------------------------------------------------------------
def plot_progression(rgb_colors: np.ndarray, cmyk: np.ndarray, out_path: str,
                      topic_labels: list[str] | None = None,
                      chunk_labels: list[str] | None = None):
    n = len(rgb_colors)
    fig_width = min(max(8, n * 0.6), 45)  # cap width so very long strips stay renderable
    dpi = 150 if n <= 60 else max(80, int(150 * 60 / n))
    fig, (ax_strip, ax_bars) = plt.subplots(
        2, 1, figsize=(fig_width, 5),
        gridspec_kw={"height_ratios": [2, 1]},
    )

    x_labels = chunk_labels if chunk_labels is not None else [f"chunk {i+1}" for i in range(n)]

    # Color strip: one rectangle per chunk
    for i, color in enumerate(rgb_colors):
        ax_strip.add_patch(Rectangle((i, 0), 1, 1, color=color))
    ax_strip.set_xlim(0, n)
    ax_strip.set_ylim(0, 1)
    ax_strip.set_yticks([])
    ax_strip.set_xticks(np.arange(n) + 0.5)
    ax_strip.set_xticklabels(x_labels, rotation=45, ha="right")
    ax_strip.set_title("Topical progression as CMYK color")

    # Stacked bar chart of the underlying topic mixture (C, M, Y, K per chunk)
    labels = topic_labels or ["Topic 1 (C)", "Topic 2 (M)", "Topic 3 (Y)", "Topic 4 (K)"]
    bottom = np.zeros(n)
    channel_colors = ["#00AEEF", "#EC008C", "#FFF200", "#333333"]  # visual proxies for C,M,Y,K
    for j in range(4):
        ax_bars.bar(np.arange(n), cmyk[:, j], bottom=bottom, color=channel_colors[j], label=labels[j])
        bottom += cmyk[:, j]
    ax_bars.set_xlim(-0.5, n - 0.5)
    ax_bars.set_ylabel("Topic weight")
    ax_bars.set_xticks(np.arange(n))
    ax_bars.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=7)
    ax_bars.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=4, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    print(f"Saved visualization to {out_path}")


# --------------------------------------------------------------------------
# 5. Main
# --------------------------------------------------------------------------
DEMO_TEXT = """
The chef selected fresh basil and ripe tomatoes for the sauce, tasting it
slowly as the garlic softened in olive oil. She adjusted the seasoning,
added a pinch of salt, and let the pasta simmer until the flavors melded
together into a rich, fragrant dish.

Meanwhile, the stock market tumbled as investors reacted to unexpected
inflation data. Analysts warned that interest rate hikes could slow
economic growth, and trading volume spiked as portfolios were
rebalanced across bonds, equities, and commodities.

The telescope captured faint light from a galaxy billions of years old,
revealing clues about the early universe. Astronomers modeled the
redshift data to estimate cosmic expansion, refining theories about dark
matter and the large-scale structure of spacetime.

Back in the kitchen, the chef plated the dish with a garnish of fresh
basil leaves and a drizzle of olive oil, tasting once more before
serving. The aroma of garlic and simmered tomatoes filled the room.
"""


def parse_df_param(value):
    """Parse a min-df/max-df CLI value: '2' -> int 2 (absolute doc count),
    '0.8' -> float 0.8 (proportion of docs), matching sklearn's own dual
    int/float convention for these parameters."""
    if value is None:
        return None
    s = str(value)
    return float(s) if "." in s else int(s)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", help="Path to a text file. Omit to use demo text.")
    parser.add_argument("--chunk-size", type=int, default=50, help="Words per chunk (default: 50)")
    parser.add_argument("--out", default="/mnt/user-data/outputs/topic_progression.png",
                         help="Output image path")
    parser.add_argument("--stopwords", choices=["english", "latin", "german", "none"], default="english",
                         help="Stopword list to use during vectorization (default: english). "
                              "Ignored for CSV input, where min-df/max-df pruning is used instead.")
    parser.add_argument("--group-col", default="doc_id",
                         help="CSV mode: column to group rows into chunks by (default: doc_id)")
    parser.add_argument("--token-col", default="token",
                         help="CSV mode: column containing the token/word text (default: token)")
    parser.add_argument("--label-col", default=None,
                         help="CSV mode: column to use for chunk labels (default: auto from full_folio, else group id)")
    parser.add_argument("--min-df", default=None,
                         help="Drop terms appearing in fewer than this many chunks (int) or this "
                              "fraction of chunks (float, e.g. 0.05). Default: 1 for text, 2 for CSV.")
    parser.add_argument("--max-df", default=None,
                         help="Drop terms appearing in more than this fraction of chunks (float) or "
                              "this many chunks (int). Default: 0.95 for text, 0.8 for CSV.")
    args = parser.parse_args()

    is_csv = bool(args.input) and args.input.lower().endswith(".csv")
    chunk_labels = None

    if is_csv:
        chunks, chunk_labels, _group_ids = chunk_by_doc_id(
            args.input, group_col=args.group_col, token_col=args.token_col, label_col=args.label_col
        )
        print(f"Grouped CSV into {len(chunks)} chunks by '{args.group_col}'.")
        # CSV tokens are already whitespace-delimited units (may include
        # digits/mixed case that are semantically meaningful, e.g. a
        # transliteration alphabet) — keep them verbatim rather than
        # re-tokenizing with a letters-only regex, and don't lowercase.
        token_pattern = r"(?u)\S+"
        lowercase = False
        stopwords = "none"
        min_df = parse_df_param(args.min_df) if args.min_df is not None else 2
        max_df = parse_df_param(args.max_df) if args.max_df is not None else 0.8
    else:
        if args.input:
            with open(args.input, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            print("No input file given — using built-in demo text.", file=sys.stderr)
            text = DEMO_TEXT

        text = re.sub(r"\s+", " ", text).strip()
        chunks = chunk_text(text, args.chunk_size)
        print(f"Split text into {len(chunks)} chunks of ~{args.chunk_size} words each.")

        token_pattern = r"(?u)\b[a-zA-ZäöüÄÖÜß]{2,}\b"
        lowercase = True
        stopwords = args.stopwords
        min_df = parse_df_param(args.min_df) if args.min_df is not None else 1
        max_df = parse_df_param(args.max_df) if args.max_df is not None else 0.95

    doc_topic, nmf, vectorizer = fit_topics(
        chunks, n_topics=N_TOPICS, stopwords=stopwords,
        min_df=min_df, max_df=max_df, token_pattern=token_pattern, lowercase=lowercase,
    )
    print(f"Vocabulary after min_df={min_df}, max_df={max_df} pruning: {len(vectorizer.get_feature_names_out())} terms")
    topics = top_words_per_topic(nmf, vectorizer)

    print("\nDiscovered topics:")
    for i, words in enumerate(topics):
        print(f"  Topic {i+1}: {', '.join(words)}")

    cmyk = topic_mixture_to_cmyk(doc_topic)
    rgb = cmyk_to_rgb(cmyk)

    print("\nPer-chunk topic mixture (= CMYK) and derived RGB:")
    for i, (mix, color) in enumerate(zip(cmyk, rgb)):
        c, m, y, k = mix
        r, g, b = (color * 255).round().astype(int)
        label = chunk_labels[i] if chunk_labels else str(i + 1)
        print(f"  {label:>8}: CMYK=({c:.2f}, {m:.2f}, {y:.2f}, {k:.2f})  RGB=({r},{g},{b})")

    topic_labels = [f"Topic {i+1}: {topics[i][0]}" for i in range(N_TOPICS)]
    plot_progression(rgb, cmyk, args.out, topic_labels=topic_labels, chunk_labels=chunk_labels)


if __name__ == "__main__":
    main()
