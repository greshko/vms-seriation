"""Diagnostics for Voynich-B-like text.

A corpus is a list of pages; a page is a list of lines; a line is a list of tokens (EVA strings).
"""
import math, random, re
from collections import Counter, defaultdict
import numpy as np
from rapidfuzz.distance import Levenshtein

import pandas as pd

GLYPH_RE = re.compile(r"c[ktpf]h|[cs]h|.")   # bench/pedestal gallows and benches as single glyphs
GALLOWS = {"k", "t", "p", "f", "ckh", "cth", "cph", "cfh"}   # plain and bench gallows (single glyphs)


def glyphs(w):
    return GLYPH_RE.findall(w)


def first_g(w):
    return glyphs(w)[0]


def last_g(w):
    return glyphs(w)[-1]


# ---------------------------------------------------------------- paths
import os as _os, json as _json
HERE = _os.path.dirname(_os.path.abspath(__file__))
CSV_NAME = "voynich_b_tokenized.csv"

def data_path(explicit=None):
    """Voynich B CSV: an explicit path, else $VMS_CSV, else ./voynich_b_tokenized.csv, else next to these scripts."""
    if explicit:
        if _os.path.exists(explicit): return explicit
        raise FileNotFoundError(f"CSV not found: {explicit}")
    for p in (_os.environ.get("VMS_CSV"), CSV_NAME, _os.path.join(HERE, CSV_NAME)):
        if p and _os.path.exists(p): return p
    raise FileNotFoundError(f"Cannot find {CSV_NAME}: put it in the working directory or next to the scripts, "
                            f"pass its path, or set the VMS_CSV environment variable.")

def load_params(path=None, **overrides):
    """Parameters from best_params.json (next to these scripts), with list-valued theme_sd turned into a tuple."""
    d = _json.load(open(path or _os.path.join(HERE, "best_params.json")))
    d.update(overrides)
    if "theme_sd" in d: d["theme_sd"] = tuple(d["theme_sd"])
    return d


# ---------------------------------------------------------------- loading
def load_real(path=None):
    df = pd.read_csv(data_path(path)).sort_values("token")
    pages = []
    for _, fdf in df.groupby("folio", sort=False):
        lines = [list(ldf.sort_values("word_line").word.astype(str)) for _, ldf in fdf.groupby("line", sort=True)]
        pages.append(lines)
    return pages


def voynich_section(folio):
    """Voynich B section of a folio label (by folio number)."""
    n = int(re.match(r"\d+", str(folio)).group())
    return ("Herbal B" if n < 57 or n in (94, 95) else "Astro/Cosmo" if n in (57, 66) else "Bio" if 75 <= n <= 84
            else "Rosette" if n in (85, 86) else "Stars" if n >= 103 else "other")


def load_layout(path=None, short=0.75):
    """Pages (as load_real), a section label per page, and inferred paragraph starts per line.
    A paragraph starts on a page's first line, or on a line containing a p/f gallows (incl. cph/cfh) that follows a line
    shorter than `short` x the page's median line length in glyphs (the 'strict' rule of the self-similarity analysis)."""
    df = pd.read_csv(data_path(path)).sort_values("token")
    pages, sections, starts = [], [], []
    for f, fdf in df.groupby("folio", sort=False):
        lines = [list(l.sort_values("word_line").word.astype(str)) for _, l in fdf.groupby("line", sort=True)]
        width = [sum(len(glyphs(w)) for w in l) + len(l) - 1 for l in lines]; med = float(np.median(width))
        st = []
        for i, l in enumerate(lines):
            pf = any(g in ("p", "f", "cph", "cfh") for w in l for g in glyphs(w))
            st.append(i == 0 or (pf and width[i - 1] < short * med))
        pages.append(lines); sections.append(voynich_section(f)); starts.append(st)
    return pages, sections, starts


def extend_layout(pages, sections, starts, n_tokens, new_sections=True):
    """Resize a layout to exactly n_tokens: truncate it, or continue it with the same pages again from the start.
    With new_sections, continued pages get new section labels ("<section> (continuation k)"), i.e. new material."""
    P, Sx, St, n, i = [], [], [], 0, 0
    while n < n_tokens:
        p = pages[i % len(pages)]; lines, st = [], []
        for li, l in enumerate(p):
            if n >= n_tokens: break
            k = min(len(l), n_tokens - n); lines.append(l[:k]); st.append(starts[i % len(pages)][li]); n += k
        rnd = i // len(pages)
        lab = sections[i % len(pages)] + (f" (continuation {rnd})" if (rnd and new_sections) else "")
        P.append(lines); Sx.append(lab); St.append(st); i += 1
    return P, Sx, St


def extend_skeleton(skeleton, sections, starts, n_tokens, new_sections=True):
    """Like extend_layout, for a skeleton (tokens per line) instead of pages of words."""
    pages = [[[None] * n for n in p] for p in skeleton]
    P, Sx, St = extend_layout(pages, sections, starts, n_tokens, new_sections)
    return [[len(l) for l in p] for p in P], Sx, St


def load_ts(path, lines_per_page=29):
    lines = [l.split() for l in open(path) if l.strip() and not l.startswith("#")]
    return [lines[i:i + lines_per_page] for i in range(0, len(lines), lines_per_page)]


def tokens(corpus):
    return [t for p in corpus for l in p for t in l]


# ---------------------------------------------------------------- information measures
def long_alpha(corpus, lo=2000, hi=50000):
    """Long-range exponent of Schinner's bit-level random walk (ASCII bits, lines joined by spaces)."""
    s = " ".join(" ".join(l) for p in corpus for l in p)
    b = np.unpackbits(np.frombuffer(s.encode("ascii"), np.uint8)).astype(np.int8) * 2 - 1
    y = np.concatenate([[0], np.cumsum(b, dtype=np.int64)])
    ls = np.unique(np.logspace(np.log10(lo), np.log10(hi), 15).astype(int))
    F = [np.sqrt((y[l:] - y[:-l]).var()) for l in ls]
    return float(np.polyfit(np.log(ls), np.log(F), 1)[0])


def mi_from_pairs(pairs):
    n = len(pairs)
    joint = Counter(pairs)
    a = Counter(x for x, _ in pairs)
    b = Counter(y for _, y in pairs)
    return sum(c / n * math.log2(c * n / (a[x] * b[y])) for (x, y), c in joint.items())


def entropy(counter):
    n = sum(counter.values())
    return -sum(c / n * math.log2(c / n) for c in counter.values())


def boundary_pairs(corpus, fa=last_g, fb=first_g):
    return [(fa(l[i]), fb(l[i + 1])) for p in corpus for l in p for i in range(len(l) - 1)]


def boundary_mi(corpus, fa=last_g, fb=first_g, null_reps=5, seed=0):
    """MI across the space, and the same after shuffling tokens within each line (null)."""
    mi = mi_from_pairs(boundary_pairs(corpus, fa, fb))
    rng = random.Random(seed)
    nulls = []
    for _ in range(null_reps):
        sh = [[rng.sample(l, len(l)) for l in p] for p in corpus]
        nulls.append(mi_from_pairs(boundary_pairs(sh, fa, fb)))
    return mi, float(np.mean(nulls))


def ngram_entropy(corpus, n):
    """Unconditional n-gram entropy of the character stream (T&S eq. 5); lines joined by spaces."""
    s = " ".join(" ".join(l) for p in corpus for l in p)
    return entropy(Counter(s[i:i + n] for i in range(len(s) - n + 1)))


def cond_entropy_h2(corpus):
    return ngram_entropy(corpus, 2) - ngram_entropy(corpus, 1)


# ---------------------------------------------------------------- lexical
def zipf_slope(toks, rmin=10, rmax=1000):
    f = np.array(sorted(Counter(toks).values(), reverse=True), float)
    r = np.arange(1, len(f) + 1)
    m = (r >= rmin) & (r <= min(rmax, len(f)))
    return float(np.polyfit(np.log(r[m]), np.log(f[m]), 1)[0])


def lex_stats(corpus, ref_vocab=None):
    toks = tokens(corpus)
    c = Counter(toks)
    L = np.array([len(glyphs(t)) for t in toks])
    out = dict(tokens=len(toks), types=len(c), ttr=len(c) / len(toks),
               hapax_frac=sum(1 for v in c.values() if v == 1) / len(c),
               len_mean=float(L.mean()), len_sd=float(L.std()),
               zipf=zipf_slope(toks))
    if ref_vocab is not None:
        out["tok_in_real_vocab"] = sum(1 for t in toks if t in ref_vocab) / len(toks)
    return out


# ---------------------------------------------------------------- positional
def jsd(p, q):
    keys = set(p) | set(q)
    P = np.array([p.get(k, 0) for k in keys], float); P /= P.sum()
    Q = np.array([q.get(k, 0) for k in keys], float); Q /= Q.sum()
    M = (P + Q) / 2
    kl = lambda A, B: np.sum(A[A > 0] * np.log2(A[A > 0] / B[A > 0]))
    return float((kl(P, M) + kl(Q, M)) / 2)


def position_stats(corpus):
    init, med, fin = Counter(), Counter(), Counter()
    init_last, med_last, fin_last = Counter(), Counter(), Counter()
    Lf, Lm, Ll = [], [], []
    for p in corpus:
        for l in p:
            if len(l) < 3:
                continue
            for i, t in enumerate(l):
                g, e, n = first_g(t), last_g(t), len(glyphs(t))
                if i == 0:
                    init[g] += 1; Lf.append(n)
                elif i == len(l) - 1:
                    fin_last[e] += 1; Ll.append(n)
                else:
                    med[g] += 1; med_last[e] += 1; Lm.append(n)
    return dict(jsd_line_initial=jsd(init, med), jsd_line_final=jsd(fin_last, med_last),
                len_first=float(np.mean(Lf)), len_mid=float(np.mean(Lm)), len_last=float(np.mean(Ll)),
                p_final_m=fin_last["m"] / sum(fin_last.values()),
                p_mid_m=med_last["m"] / max(1, sum(med_last.values())))


# ---------------------------------------------------------------- sequential / local similarity
def adjacency_stats(corpus):
    same = ed1 = n = 0
    ed_adj = []
    for p in corpus:
        for l in p:
            for a, b in zip(l, l[1:]):
                d = Levenshtein.distance(a, b)
                n += 1; same += d == 0; ed1 += d == 1; ed_adj.append(d)
    return dict(p_repeat=same / n, p_ed1=ed1 / n, ed_adjacent=float(np.mean(ed_adj)))


def mbar_curve(corpus, maxd=20, max_pairs_per_d=40000, seed=0):
    """T&S Fig. 4: mean edit distance of token pairs as a function of line distance (within page)."""
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for p in corpus:
        for i in range(len(p)):
            for j in range(i, min(len(p), i + maxd + 1)):
                for a in p[i]:
                    for b in p[j]:
                        if i == j and a is b:
                            continue
                        buckets[j - i].append((a, b))
    out = []
    for d in range(maxd + 1):
        pairs = buckets[d]
        if len(pairs) > max_pairs_per_d:
            pairs = rng.sample(pairs, max_pairs_per_d)
        out.append(float(np.mean([Levenshtein.distance(a, b) for a, b in pairs])) if pairs else np.nan)
    return np.array(out)


def global_mbar(corpus, n=60000, seed=1):
    rng = random.Random(seed)
    t = tokens(corpus)
    return float(np.mean([Levenshtein.distance(rng.choice(t), rng.choice(t)) for _ in range(n)]))


def page_burstiness(corpus, minc=15):
    """Mean over frequent types of (observed pages containing type)/(expected under random placement).
    <1 means the type clusters on few pages (the 'contextual vocabulary' effect)."""
    page_tok = [[t for l in p for t in l] for p in corpus]
    sizes = np.array([len(x) for x in page_tok], float)
    N = sizes.sum()
    c = Counter(t for x in page_tok for t in x)
    present = Counter(t for x in page_tok for t in set(x))
    ratios = []
    for w, k in c.items():
        if k < minc:
            continue
        exp = np.sum(1 - (1 - sizes / N) ** k)
        ratios.append(present[w] / exp)
    return float(np.mean(ratios))


def neighbour_frac(corpus, minc=2):
    """Fraction of types (count>=minc) having an edit-distance-1 neighbour among types (count>=minc)."""
    c = Counter(tokens(corpus))
    V = [w for w, k in c.items() if k >= minc]
    Vs = set(V)
    alpha = sorted(set("".join(V)))
    def has_nb(w):
        for i in range(len(w) + 1):
            for a in alpha:
                if w[:i] + a + w[i:] in Vs: return True
                if i < len(w) and a != w[i] and w[:i] + a + w[i + 1:] in Vs: return True
            if i < len(w) and w[:i] + w[i + 1:] in Vs: return True
        return False
    return sum(has_nb(w) for w in V) / len(V)


# ---------------------------------------------------------------- all together
def summarize(corpus, ref_vocab=None, curve=True):
    s = {}
    s.update(lex_stats(corpus, ref_vocab))
    s["H1"] = ngram_entropy(corpus, 1); s["H2"] = ngram_entropy(corpus, 2); s["H3"] = ngram_entropy(corpus, 3)
    s["h2_cond"] = s["H2"] - s["H1"]
    mi, null = boundary_mi(corpus)
    s["MI_boundary"] = mi; s["MI_boundary_null"] = null; s["MI_boundary_excess"] = mi - null
    mi2, null2 = boundary_mi(corpus, fa=lambda w: w[-2:], fb=lambda w: w[:2], null_reps=3)
    s["MI_bound2_excess"] = mi2 - null2
    s.update(position_stats(corpus))
    s.update(adjacency_stats(corpus))
    s["burstiness"] = page_burstiness(corpus)
    s["ed1_neighbour_frac"] = neighbour_frac(corpus)
    if curve:
        m = mbar_curve(corpus)
        s["Mbar0"], s["Mbar1"], s["Mbar20"] = m[0], m[1], m[20]
        s["Mbar_global"] = global_mbar(corpus)
        s["_curve"] = m
    return s
