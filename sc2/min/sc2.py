"""SC2: a boundary- and position-aware self-citation generator for Voynich-B-like text.

No cipher, no plaintext. Every token is still produced by copying a token written earlier
(mostly on the same page, preferring nearby lines) and optionally mutating it, as in
Timm & Schinner (2019). What changes is how the scribe *chooses* among options:

  1. propose K candidates  (source selection x mutation operator)
  2. score each candidate  [shape(w) > floor]                  word-internal well-formedness gate
                           x [P*(first|prev last)/P_recent(first|prev last)]^beta  flow across the space
                           x [P*(first|pos)/P_recent(first|pos)]^gamma  pos = initial/medial/final
                           x [P*(last|pos)/P_recent(last|pos)]^delta
  3. sample one in proportion to its score.

Plus a 'span copy' operator: continue copying the token that followed the previous
source in the source line, so boundary pairs propagate by citation, not only by scoring.

The three affinity tables (A_bound, A_start, A_end) and the glyph-trigram shape model are
estimated from a reference corpus; everything else is a handful of scalar parameters.
"""
import math, random
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from metrics import glyphs, first_g, last_g, GALLOWS
from rapidfuzz.distance import Levenshtein as _Lev

# glyph similarity classes (after Timm's 'voynich_similarities' + common e/i run edits)
SIM = [set("oay"), set("dls"), set("nrlm"), set("rs"), set("ktpf"), set("oe"), set("mg"),
       {"ch", "sh"}, {"ckh", "cth", "cph", "cfh"}, {"k", "ckh"}, {"t", "cth"}]
SIMMAP = defaultdict(set)
for cls in SIM:
    for g in cls:
        SIMMAP[g] |= cls - {g}
PREFIXES = [tuple(glyphs(p)) for p in ["qo", "o", "d", "ch", "sh", "y", "s", "l", "qok", "ok", "ot", "qot"]]
def _starts(g, p): return tuple(g[:len(p)]) == p and len(g) > len(p) + 1


# ------------------------------------------------------------------ learned components
def pos_class(i, n):
    """Line-position class: I (first), 2 (second, lines of 4+), F (last, lines of 3+), M (other)."""
    if i == 0: return "I"
    if i == n - 1 and n >= 3: return "F"
    if i == 1 and n >= 4: return "2"
    return "M"


import re as _re
_FAMILY = _re.compile(r"^(qo|o)[kt]e{1,3}(?!e)")
THEME_FEATURES = ("family", "gallows", "y", "lr", "length")
def theme_features(w):
    """Features a line theme can favour: qok/ot + e-run family, contains a gallows, ends -y, ends -l/-r, (standardised) length."""
    g = glyphs(w)
    return (1.0 if _FAMILY.match(w) and len(w) > _FAMILY.match(w).end() else 0.0, 1.0 if any(x in GALLOWS for x in g) else 0.0,
            1.0 if g[-1] == "y" else 0.0, 1.0 if g[-1] in ("l", "r") else 0.0, (len(g) - 4.7) / 1.5)


CHUNK_RX = _re.compile(r"^(?P<pre>(?:qo|q|o|y|d|s(?!h)|l|r)?(?:k|t|p|f)?)(?P<mid>(?:ckh|cth|cph|cfh|ch|sh)?e{0,3})(?P<suf>.*)$")
def chunks(w):
    m = CHUNK_RX.match(w); return m["pre"], m["mid"], m["suf"]


LEN_CAP = 10
def len_bin(w): return min(len(glyphs(w)), LEN_CAP)


class ShapeModel:
    """Glyph trigram model over words (with ^ $ boundaries), interpolated with bigrams."""
    def __init__(self, words, lam=(0.7, 0.25, 0.05)):
        self.t3, self.c2, self.b2, self.c1, self.u = Counter(), Counter(), Counter(), Counter(), Counter()
        for w in words:
            g = ["^", "^"] + glyphs(w) + ["$"]
            for i in range(2, len(g)):
                self.t3[tuple(g[i-2:i+1])] += 1; self.c2[tuple(g[i-2:i])] += 1
                self.b2[tuple(g[i-1:i+1])] += 1; self.c1[g[i-1]] += 1; self.u[g[i]] += 1
        self.N = sum(self.u.values()); self.V = len(self.u) + 1; self.lam = lam
        self.cache = {}

    def to_json(self):
        pack = lambda c: [list(k) + [v] if isinstance(k, tuple) else [k, v] for k, v in c.items()]
        return dict(lam=list(self.lam), N=self.N, V=self.V, t3=pack(self.t3), c2=pack(self.c2), b2=pack(self.b2), c1=pack(self.c1), u=pack(self.u))

    @classmethod
    def from_json(cls, d):
        m = cls([], tuple(d["lam"]))
        unpack = lambda rows: Counter({(tuple(r[:-1]) if len(r) > 2 else r[0]): r[-1] for r in rows})
        m.t3, m.c2, m.b2, m.c1, m.u = (unpack(d[k]) for k in ("t3", "c2", "b2", "c1", "u"))
        m.N, m.V = d["N"], d["V"]
        return m

    def logp(self, w):
        if w in self.cache:
            return self.cache[w]
        g = ["^", "^"] + glyphs(w) + ["$"]
        s, illegal = 0.0, False
        for i in range(2, len(g)):
            a, b, c = g[i-2], g[i-1], g[i]
            p3 = self.t3[(a, b, c)] / self.c2[(a, b)] if self.c2[(a, b)] else 0
            p2 = self.b2[(b, c)] / self.c1[b] if self.c1[b] else 0
            p1 = (self.u[c] + 1) / (self.N + self.V)
            if self.b2[(b, c)] == 0:
                illegal = True
            s += math.log(self.lam[0]*p3 + self.lam[1]*p2 + self.lam[2]*p1)
        r = (s / (len(g) - 2), illegal)
        self.cache[w] = r
        return r


class CondTable:
    """P(y|x) additively smoothed toward the marginal P(y). Callable as table(x, y); stores only counts, so it can be saved."""
    def __init__(self, pairs=(), smooth=3.0):
        pairs = list(pairs)
        self.joint = Counter(pairs); self.cx = Counter(x for x, _ in pairs); self.cy = Counter(y for _, y in pairs)
        self.n = len(pairs); self.smooth = smooth

    def __call__(self, x, y):
        py = (self.cy[y] + 0.5) / (self.n + 0.5 * len(self.cy))
        return (self.joint[(x, y)] + self.smooth * py) / (self.cx[x] + self.smooth)

    def to_json(self):
        return dict(smooth=self.smooth, n=self.n, joint=[[x, y, c] for (x, y), c in self.joint.items()],
                    cx=[[x, c] for x, c in self.cx.items()], cy=[[y, c] for y, c in self.cy.items()])

    @classmethod
    def from_json(cls, d):
        t = cls((), d["smooth"]); t.n = d["n"]
        t.joint = Counter({(x, y): c for x, y, c in d["joint"]}); t.cx = Counter({x: c for x, c in d["cx"]}); t.cy = Counter({y: c for y, c in d["cy"]})
        return t


def cond_table(pairs, smooth=3.0):
    return CondTable(pairs, smooth)


def fit_tables(real_corpus):
    """Targets the scribe steers toward: P(first|prev last), P(first|line-initial), P(last|line-final),
    P(first|medial), P(last|medial), plus the word-shape gate."""
    words = [t for p in real_corpus for l in p for t in l]
    bpairs, fpairs, lpairs, npairs = [], [], [], []
    for p in real_corpus:
        for l in p:
            for a, b in zip(l, l[1:]):
                bpairs.append((last_g(a), first_g(b)))
            for i, t in enumerate(l):
                pos = pos_class(i, len(l))
                fpairs.append((pos, first_g(t))); lpairs.append((pos, last_g(t))); npairs.append((pos, len_bin(t)))
    fc = Counter(g for _, g in fpairs); lc = Counter(g for _, g in lpairs)
    # stray standalone h (a writing slip): rate after each preceding glyph, and initial ch/sh -> h
    hb, hn = Counter(), Counter()
    for w in words:
        g = glyphs(w)
        for a, b in zip(g, g[1:]):
            if b == "h": hb[a] += 1
        for a in (g[:-1] if g[-1] == "h" else g): hn[a] += 1
    h_after = {a: hb[a] / hn[a] for a in hb if a != "h" or hn[a]}
    n_bench_init = sum(glyphs(w)[0] in ("ch", "sh") for w in words)
    h_init = sum(glyphs(w)[0] == "h" for w in words) / max(1, n_bench_init)
    fm = Counter(b for _, b in bpairs); nb = sum(fm.values())
    first_marg = {b: v / nb for b, v in fm.items()}
    ch = {"P": set(), "C": set(), "S": set()}
    for w in words:
        for k, x in zip("PCS", chunks(w)):
            if x: ch[k].add(x)
    return dict(first_marg=first_marg, chunks=ch, h_after=h_after, h_init=h_init, first_alpha=[g for g, c in fc.items() if c >= 20], last_alpha=[g for g, c in lc.items() if c >= 20],
                shape=ShapeModel(words), bound=cond_table(bpairs),
                first=cond_table(fpairs), last=cond_table(lpairs), length=cond_table(npairs))


MODEL_VERSION = 1
_COND = ("bound", "first", "last", "length")


def tables_to_json(T):
    return dict(shape=T["shape"].to_json(), **{k: T[k].to_json() for k in _COND},
                first_marg=T["first_marg"], chunks={k: sorted(v) for k, v in T["chunks"].items()}, h_after=T["h_after"], h_init=T["h_init"],
                first_alpha=T["first_alpha"], last_alpha=T["last_alpha"])


def tables_from_json(d):
    return dict(shape=ShapeModel.from_json(d["shape"]), **{k: CondTable.from_json(d[k]) for k in _COND},
                first_marg=d["first_marg"], chunks={k: set(v) for k, v in d["chunks"].items()}, h_after=d["h_after"], h_init=d["h_init"],
                first_alpha=d["first_alpha"], last_alpha=d["last_alpha"])


def save_model(path, tables, skeleton, sections=None, para_starts=None, source=""):
    """Everything generation needs, without the source text: fitted tables (counts only) and the layout
    (tokens per line, a section per page, paragraph starts per line)."""
    import json
    json.dump(dict(version=MODEL_VERSION, source=source, tables=tables_to_json(tables), skeleton=skeleton,
                   sections=sections, para_starts=para_starts), open(path, "w"))


def load_model(path):
    """Returns (tables, skeleton, sections, para_starts)."""
    import json
    d = json.load(open(path))
    if d.get("version") != MODEL_VERSION: raise ValueError(f"{path}: unsupported model version {d.get('version')}")
    return tables_from_json(d["tables"]), d["skeleton"], d["sections"], d["para_starts"]


class Tracker:
    """Running (exponentially forgotten) conditional frequencies of the scribe's own output."""
    def __init__(self, halflife=1500):
        self.c = defaultdict(Counter); self.tot = Counter(); self.k = 0.5 ** (1 / halflife)
    def add(self, x, y):
        C = self.c[x]
        for key in C: C[key] *= self.k
        self.tot[x] = self.tot[x] * self.k + 1; C[y] += 1
    def p(self, x, y, prior):
        # shrink toward the target prior when little evidence has accumulated
        return (self.c[x][y] + 20 * prior) / (self.tot[x] + 20)


# ------------------------------------------------------------------ generator
@dataclass
class Params:
    K: int = 10                 # candidates considered per token
    tau: float = 3.0            # recency decay (lines) for source choice
    p_same_pos: float = 0.15    # take the token at the same position in the previous line
    p_span: float = 0.20        # continue copying the source line (neighbour of last source)
    p_global: float = 0.05      # draw a source from anywhere in earlier text
    op_w: dict = field(default_factory=lambda: dict(copy=0.12, sub=0.45, prefix=0.18, combine=0.08, split=0.09, erun=0.08))
    len_max: int = 10           # hard crop, as in T&S
    p_cross_class: float = 0.25 # chance to cite across position classes (initial/medial/final)
    shape_floor: float = -2.3   # reject words whose mean glyph log-prob falls below this (a gate, not a preference)
    beta: float = 1.0           # boundary flow weight
    gamma: float = 1.0          # line-initial weight
    delta: float = 1.0          # line-final weight
    p_para_gallows: float = 0.9 # paragraph-initial gallows (first token of page)
    no_immediate_repeat: float = 0.3  # multiplicative penalty for writing the previous token again
    p_extra_mut: float = 0.3    # chance of a further mutation (geometric)
    p_flow: float = 0.3         # chance a candidate's start is re-shaped to follow the previous ending
    p_flow_pos: float = 0.3     # same for line-initial starts / line-final endings
    p_concat: float = 0.0       # chance a candidate is a whole-word compound (cited word + a whole recent word)
    concat_edge_boost: float = 2.0  # compounds are this much likelier at line starts and ends
    concat_max: int = 12        # length cap for compounds (their halves have already passed the gate)
    p_multi: float = 0.0        # chance a candidate gets a 2-3 edit mutation at once
    bypass_w: float = 0.0       # weight for a gate-failing candidate whose prefix/core/suffix all occur in B
    bound_lift: float = 0.0     # direct boundary preference: (P*(first|prev last) / P*(first))^bound_lift
    p_pair: float = 0.0         # pair copying: after a word, write a remembered successor of it (a word that followed it before)
    pair_gamma: float = 1.5     # remembered successors weighted by (times the pair was written)^pair_gamma
    p_dup: float = 0.0          # reduplication: write the previous word again; rate = p_dup * (per-mille frequency so far)^dup_exp
    dup_exp: float = 0.7
    dup_first_pair: float = 0.3 # relative reduplication rate for the first pair of a line
    dup_last_pair: float = 0.5  # ... and for the last pair
    no_near_repeat: float = 1.0 # weight for a candidate one edit away from the previous word
    p_section: float = 0.0      # chance a citation comes from the section's running pool (shared vocabulary across pages)
    sec_scale: float = 0.0      # mean look-back (tokens) into the section pool; 0 = uniform over the section so far
    para_reset: bool = True
    p_para_leak: float = 0.0    # chance a page-local citation may reach back past the paragraph start (partial reset)     # page-local routes stay inside the current paragraph (needs paragraph starts)
    p_freq: float = 0.0         # chance a global-memory draw is a frequency draw from the commonest words so far
    freq_gamma: float = 1.5     # frequency draw weights count^gamma (gamma > 1: more than proportional)
    freq_top: int = 200         # size of the "commonest words" list
    freq_in_section: bool = False
    p_freq_verbatim: float = 0.0    # chance a frequency-drawn (well-known) word is written without mutation   # also apply frequency draws to the section pool (with the section's own counts)
    fam_bins: tuple = (1, 2, 3, 5, 10, 20, 50, 100)   # upper bounds of prior-count bins (a last bin catches the rest)
    fam_mult: tuple = ()
    fam_scope: str = "global"   # "section": familiarity and the frequency memory count occurrences within the current section        # graded familiarity: candidate weight by how often the form has been written so far (empty = off)
    p_oneoff_skip: float = 0.9  # chance a citation that lands on a one-off form is redrawn
    p_single_skip: float = 0.0  # chance a citation that lands on any form written only once is redrawn (familiarity bias)
    theme_sd: tuple = (0.0, 0.0, 0.0, 0.0, 0.0)   # spread of the per-line preference for each THEME_FEATURES entry (0 = off)
    p_anchor: float = 0.0       # chance a token is cited from the line's anchor (model) line
    tau_anchor: float = 4.0     # recency scale (lines) for choosing the anchor line on the page
    lam_len: float = 2.0        # strength of corrective word-length steering by line position
    p_slip_copy: float = 0.5    # chance that citing a slipped occurrence copies the slip
    h_scale: float = 1.0
    h_init_scale: float = 1.0   # extra factor on the word-initial (ch/sh -> h) slip rate        # multiplier on B's standalone-h slip rates (0 = off)
    max_rounds: int = 4         # rounds of K candidates before the verbatim fallback
    global_scale: float = 0.0   # mean look-back (tokens) of long-memory citation; 0 = uniform over history
    seed: int = 0


class SC2:
    def __init__(self, tables, params: Params, flags=None):
        self.T = tables; self.P = params
        self.rng = random.Random(params.seed)
        # ablation switches
        self.flags = dict(bound=True, start=True, end=True, span=True, shape=True, strat=True, flow=True)
        self.FIRST = tables["first_alpha"]; self.LAST = tables["last_alpha"]
        self._cls = "M"
        self.trk_b, self.trk_f, self.trk_l, self.trk_n = Tracker(), Tracker(), Tracker(), Tracker()
        self.n_written, self.slips = Counter(), defaultdict(Counter)   # per intended word: times written, slip forms
        self.seen_written = set(); self._anchor = None; self._theme = None; self.oneoff = set()
        self._lo = 0; self._lo_page = 0; self._sec = "all"; self.sec_hist = defaultdict(list)
        self.sec_count = defaultdict(Counter); self._freq_cache = {}; self._n_total = 0; self._last_freq = False
        self.pair_mem = defaultdict(Counter)                           # word -> Counter of words written right after it (same line)
        self.stats = Counter()
        if flags: self.flags.update(flags)

    # ---- mutation operators (all glyph-level, order-preserving as in T&S)
    def op_sub(self, w):
        g = glyphs(w); idx = [i for i, x in enumerate(g) if x in SIMMAP]
        if not idx: return w
        i = self.rng.choice(idx); g[i] = self.rng.choice(sorted(SIMMAP[g[i]]))
        return "".join(g)

    def op_prefix(self, w):
        g = glyphs(w); pre = [p for p in PREFIXES if _starts(g, p)]
        if pre and self.rng.random() < 0.5:
            return "".join(g[len(max(pre, key=len)):])
        if pre: return w
        return "".join(self.rng.choice(PREFIXES)) + w

    def op_combine(self, w, pool):
        o = self.rng.choice(pool)[0]; go, gw = glyphs(o), glyphs(w)
        if len(go) < 2 or len(gw) < 2: return w
        return "".join(gw[:self.rng.randint(1, len(gw) - 1)] + go[self.rng.randint(1, len(go) - 1):])

    def op_split(self, w):
        g = glyphs(w)
        if len(g) < 5: return w
        k = self.rng.randint(2, len(g) - 2)
        return "".join(g[:k] if self.rng.random() < 0.5 else g[k:])

    def op_erun(self, w):
        g = glyphs(w)
        runs = [i for i, x in enumerate(g) if x in ("e", "i")]
        if not runs: return w
        i = self.rng.choice(runs)
        if self.rng.random() < 0.5 and (i + 1 < len(g) and g[i+1] == g[i] or i > 0 and g[i-1] == g[i]):
            del g[i]
        else:
            g.insert(i, g[i])
        return "".join(g)

    # ---- context-driven edits (Currier's 'first glyph suggested by the previous last glyph')
    def _sample_target(self, table, x, alphabet):
        ws = [table(x, g) for g in alphabet]
        return self.rng.choices(alphabet, ws)[0]

    def flow_start(self, w, target):
        g = glyphs(w)
        if g[0] == target: return w
        opts = []
        if len(g) > 2: opts.append(target + "".join(g[1:]))          # swap the first glyph
        opts.append(target + w)                                      # prepend
        if target == "q": opts.append("q" + w if g[0] == "o" else "qo" + w)
        for p in PREFIXES:                                           # swap a prefix
            if _starts(g, p): opts.append(target + "".join(g[len(p):]))
        return self.rng.choice(opts)

    def flow_end(self, w, target):
        g = glyphs(w)
        if g[-1] == target: return w
        if len(g) > 2: return "".join(g[:-1]) + target              # swap the last glyph (length-neutral)
        return w + target                                            # very short words: append

    def _freq_draw(self, counts, key):
        """Draw among the commonest words so far with weight count^gamma (lists refreshed every 100 writes)."""
        cache = self._freq_cache.get(key)
        if cache is None or self._n_total - cache[0] >= 100:
            top = counts.most_common(self.P.freq_top)
            if not top: return None
            cache = (self._n_total, [w for w, _ in top], [c ** self.P.freq_gamma for _, c in top]); self._freq_cache[key] = cache
        self.stats["freq_draw"] += 1
        return self.rng.choices(cache[1], cache[2])[0]

    def _global(self, history):
        self._last_freq = False
        if self.P.p_freq and self.rng.random() < self.P.p_freq:
            w = self._freq_draw(self.sec_count[self._sec] if self.P.fam_scope == "section" else self.n_written,
                                ("section", self._sec) if self.P.fam_scope == "section" else "global")
            if w is not None: self._last_freq = True; return w
        # long memory: exponentially recency-weighted over the whole manuscript so far (0 = uniform)
        if not self.P.global_scale:
            return self.rng.choice(history)
        n = len(history); lam = 1 / self.P.global_scale
        # truncated exponential on [0, n): look-backs never exceed the text written so far
        u = self.rng.random(); x = -math.log(1 - u * (1 - math.exp(-lam * n))) / lam
        return history[-(min(int(x), n - 1) + 1)]

    def pick_anchor(self, src_page, line_no):
        """At a line start, choose an earlier line (recency-weighted) as the model for this line."""
        cands = [li for li in range(self._lo, line_no) if src_page[li]]
        if not cands: self._anchor = None; return
        w = [math.exp(-(line_no - li) / self.P.tau_anchor) for li in cands]
        self._anchor = self.rng.choices(cands, w)[0]

    def _section_draw(self, history):
        pool = self.sec_hist.get(self._sec) or []
        if not pool: return self._global(history), None, "global"
        if self.P.freq_in_section and self.P.p_freq and self.rng.random() < self.P.p_freq:
            w = self._freq_draw(self.sec_count[self._sec], ("section", self._sec))
            if w is not None: return w, None, "freq"
        n = len(pool)
        if not self.P.sec_scale: return pool[self.rng.randrange(n)], None, "section"
        lam = 1 / self.P.sec_scale; u = self.rng.random(); x = -math.log(1 - u * (1 - math.exp(-lam * n))) / lam
        return pool[-(min(int(x), n - 1) + 1)], None, "section"

    # ---- source selection
    def choose_source(self, page, line_no, pos, prev_src, history):
        for _ in range(4):                                            # one-off forms are rarely re-cited
            t, loc, how = self._choose_source(page, line_no, pos, prev_src, history)
            if t in self.oneoff:
                if self.rng.random() >= self.P.p_oneoff_skip: return t, loc, how
                self.stats["oneoff_redrawn"] += 1; continue
            if self.n_written[t] == 1 and self.rng.random() < self.P.p_single_skip:
                self.stats["single_redrawn"] += 1; continue
            return t, loc, how
        return t, loc, how

    def _choose_source(self, page, line_no, pos, prev_src, history):
        lo = self._lo                                                  # first line of the current paragraph (page-local routes stay after it)
        if lo > self._lo_page and self.rng.random() < self.P.p_para_leak: lo = self._lo_page   # partial reset: sometimes look past it
        if self._anchor is not None and lo <= self._anchor < line_no and page[self._anchor] and self.rng.random() < self.P.p_anchor:
            ti = self.rng.randrange(len(page[self._anchor])); return page[self._anchor][ti], (self._anchor, ti), "anchor"
        r = self.rng.random()
        if self.flags["span"] and prev_src is not None and r < self.P.p_span:
            (sl, si) = prev_src
            if sl >= lo and si + 1 < len(page[sl]):
                return page[sl][si + 1], (sl, si + 1), "span"
        r = self.rng.random()
        local_empty = not any(page[lo:line_no + 1])
        if r < self.P.p_same_pos:
            if line_no - 1 >= lo and pos < len(page[line_no - 1]):
                return page[line_no - 1][pos], (line_no - 1, pos), "pos"
            # no token above in this paragraph: fall through, leaving the other rates untouched
        elif r < self.P.p_same_pos + self.P.p_global:
            w = self._global(history); return w, None, ("freq" if self._last_freq else "global")
        elif r < self.P.p_same_pos + self.P.p_global + self.P.p_section:
            return self._section_draw(history)
        if local_empty:                                                # paragraph start: draw from the section stock
            return self._section_draw(history) if self.P.p_section or self.sec_hist.get(self._sec) else (self._global(history), None, "global")
        # recency-weighted token from this paragraph (this page when there are no paragraph marks)
        cands, wts = [], []
        want = self._cls
        strat = self.flags["strat"] and self.rng.random() > self.P.p_cross_class
        for li in range(lo, line_no + 1):
            wl = math.exp(-(line_no - li) / self.P.tau)
            L = page[li]; C = self._page_cls[li] if li < len(self._page_cls) else ["M"] * len(L)
            for ti, t in enumerate(L):
                c = C[ti] if ti < len(C) else "M"
                if strat and c != want: continue
                cands.append((t, (li, ti))); wts.append(wl)
        if not cands:
            return self._global(history), None, "global"
        t, loc = self.rng.choices(cands, wts)[0]
        return t, loc, "page"

    def op_concat(self, w, pool):
        """Whole-word compound: the cited word joined to a whole recent word. A part ending in -l/-o/-r goes first."""
        o = self.rng.choice(pool)
        if o == w: return w
        a, b = (w, o) if glyphs(w)[-1] in ("l", "o", "r") else ((o, w) if glyphs(o)[-1] in ("l", "o", "r") else (w, o))
        return a + b

    def op_multi(self, w, pool):
        """Two or three edits at once (never a plain copy): produces forms far from any recurring word."""
        ops = [o for o in self.P.op_w if o != "copy"]; wts = [self.P.op_w[o] for o in ops]
        for _ in range(self.rng.choice((2, 3))):
            op = self.rng.choices(ops, wts)[0]
            w = {"sub": lambda x: self.op_sub(x), "prefix": lambda x: self.op_prefix(x), "combine": lambda x: self.op_combine(x, pool),
                 "split": lambda x: self.op_split(x), "erun": lambda x: self.op_erun(x)}[op](w)
        return w

    def chunk_attested(self, w):
        return all(not x or x in self.T["chunks"][k] for k, x in zip("PCS", chunks(w)))

    def mutate(self, w, pool):
        w = self._mutate1(w, pool)
        while self.rng.random() < self.P.p_extra_mut:
            w = self._mutate1(w, pool)
        return w

    def _mutate1(self, w, pool):
        ops = list(self.P.op_w); op = self.rng.choices(ops, [self.P.op_w[o] for o in ops])[0]
        if op == "copy": return w
        if op == "sub": return self.op_sub(w)
        if op == "prefix": return self.op_prefix(w)
        if op == "combine": return self.op_combine(w, pool)
        if op == "split": return self.op_split(w)
        return self.op_erun(w)

    def score(self, w, prev, cls, override=None):
        """override: None = normal gate; a number = skip the gate/length cap and multiply by it (compounds, bypass)."""
        s = 1.0 if override is None else override
        n = len(glyphs(w))
        if override is None:
            if self.flags["shape"] and w not in self.seen_written:      # anything already written counts as legal
                lp, illegal = self.T["shape"].logp(w)
                if illegal or lp < self.P.shape_floor: return 0.0
            if n < 2 and w not in ("s", "y", "r", "l", "o", "d"): return 0.0
            if n > self.P.len_max: return 0.0
        if s == 0: return 0.0
        if self.P.fam_mult:                                          # graded familiarity: re-use depends on how established a form is
            k = (self.sec_count[self._sec] if self.P.fam_scope == "section" else self.n_written).get(w, 0)
            if k:
                i = next((j for j, ub in enumerate(self.P.fam_bins) if k <= ub), len(self.P.fam_bins))
                s *= self.P.fam_mult[min(i, len(self.P.fam_mult) - 1)]
        f, l = first_g(w), last_g(w)
        # corrective weights: target / what I have recently been writing in this context
        if prev is not None and self.flags["bound"]:
            a = last_g(prev); tgt = self.T["bound"](a, f)
            s *= (tgt / self.trk_b.p(a, f, tgt)) ** self.P.beta
            if self.P.bound_lift:
                s *= (tgt / self.T["first_marg"].get(f, 1e-4)) ** self.P.bound_lift
        if self.flags["start"]:
            tgt = self.T["first"](cls, f); s *= (tgt / self.trk_f.p(cls, f, tgt)) ** self.P.gamma
        if self.flags["end"]:
            tgt = self.T["last"](cls, l); s *= (tgt / self.trk_l.p(cls, l, tgt)) ** self.P.delta
        if self.P.lam_len:
            b = len_bin(w); tgt = self.T["length"](cls, b); s *= (tgt / self.trk_n.p(cls, b, tgt)) ** self.P.lam_len
        if prev is not None and w == prev:
            s *= self.P.no_immediate_repeat
        elif prev is not None and self.P.no_near_repeat != 1.0 and abs(len(w) - len(prev)) <= 3 and _Lev.distance(w, prev) == 1:
            s *= self.P.no_near_repeat
        if self._theme is not None:                                  # line theme: this line's leanings
            s *= math.exp(sum(z * x for z, x in zip(self._theme, theme_features(w))))
        return s

    def legal(self, w):
        n = len(glyphs(w))
        if n > self.P.len_max or (n < 2 and w not in ("s", "y", "r", "l", "o", "d")): return False
        if not self.flags["shape"]: return True
        lp, illegal = self.T["shape"].logp(w); return not illegal and lp >= self.P.shape_floor

    def page_gallows(self, w):
        """Paragraph/page-initial gallows (T&S): unless the word already opens with a gallows or bench-gallows glyph,
        prepend a plain gallows (chedy -> pchedy); if no prepended form passes the shape gate, replace a non-bench
        opening glyph instead (daiin -> paiin, qokedy -> pokedy). Benches are never split."""
        g = glyphs(w)
        if self.rng.random() >= self.P.p_para_gallows or g[0] in GALLOWS: return w
        opts = [(x, "prepend") for x in ["p", "t", "k", "f"]]
        if g[0] not in ("ch", "sh") and len(g) >= 3: opts += [(x, "replace") for x in ["p", "t", "k", "f"]]
        self.rng.shuffle(opts); opts.sort(key=lambda o: o[1] != "prepend")    # prefer prepending, then replace a non-bench first glyph
        for x, how in opts:
            v = x + w if how == "prepend" else x + "".join(g[1:])
            if self.legal(v): return v
        return w

    def stray_h(self, w):
        """Rare writing slip producing a standalone h glyph, at Voynich-B rates: a word-initial ch/sh may lose its
        first stroke (chy -> hy), and an h may follow a glyph with the probability B shows after that glyph
        (mostly after gallows and bench-gallows: cthy -> cthhy). Benches themselves are never split.
        Slips are written only; the scribe keeps citing the intended form, so slips never propagate."""
        g = glyphs(w); sc = self.P.h_scale
        if g[0] in ("ch", "sh") and self.rng.random() < self.T["h_init"] * sc * self.P.h_init_scale:
            g[0] = "h"; self.stats["h_initial"] += 1
        out = []
        for x in g:
            out.append(x)
            if self.rng.random() < self.T["h_after"].get(x, 0.0) * sc:
                out.append("h"); self.stats["h_after"] += 1
        return "".join(out)

    # ---- main loop
    def generate(self, skeleton, seed_line, sections=None, para_starts=None):
        """skeleton: pages -> line lengths (tokens per line).
        sections: optional section label per page (enables the section citation pool).
        para_starts: optional per page, per line, True where a paragraph begins (enables paragraph resets)."""
        out, history = [], list(seed_line)
        prev_page = [seed_line]; prev_cls = [[pos_class(i, len(seed_line)) for i in range(len(seed_line))]]
        self.sec_hist = defaultdict(list); self.sec_count = defaultdict(Counter); self._freq_cache = {}; self._n_total = 0
        self.pair_mem = defaultdict(Counter)
        for pi, pg_lens in enumerate(skeleton):
            self._sec = sections[pi] if sections else "all"
            starts = para_starts[pi] if (para_starts and self.P.para_reset) else [i == 0 for i in range(len(pg_lens))]
            pst = 0
            page, written = [], []          # page: intended forms (the citation pool); written: what appears on the page
            page_cls = []                   # position class of every intended token (for stratified citation)
            for li, n in enumerate(pg_lens):
                if li > 0 and starts[li]: pst = li                    # a new paragraph begins on this line
                page.append([]); written.append([]); page_cls.append([])
                prev_src = None
                for pos in range(n):
                    # early in a page: allow citing from the previous page (T&S footnote 8)
                    src_page = page if sum(map(len, page)) >= 6 else prev_page + page
                    self._page_cls = page_cls if src_page is page else prev_cls + page_cls
                    li_eff = li if src_page is page else li + len(prev_page)
                    # page-local citation stays inside the paragraph; without paragraph marks the old page-start overlap is kept
                    self._lo = (pst if src_page is page else len(prev_page) + pst) if (para_starts and self.P.para_reset) else 0
                    self._lo_page = 0                                      # a leak may reach earlier lines of the page (or, at a page start, the previous page's last lines)
                    prev = page[li][-1] if pos > 0 else None
                    cls = pos_class(pos, n); self._cls = cls
                    if pos == 0:
                        self.pick_anchor(src_page, li_eff)
                        self._theme = [self.rng.gauss(0, sd) for sd in self.P.theme_sd] if any(self.P.theme_sd) else None
                    initial, final = cls == "I", cls == "F"
                    cands, wts, routes = [], [], []
                    dup_word = None                                          # reduplication of the previous word
                    if prev is not None and self.P.p_dup and history:
                        pm = 1000.0 * self.n_written[prev] / len(history)
                        pr = self.P.p_dup * pm ** self.P.dup_exp * (self.P.dup_first_pair if pos == 1 else self.P.dup_last_pair if pos == n - 1 else 1.0)
                        if self.rng.random() < pr: dup_word = prev; self.stats["reduplicated"] += 1
                    if dup_word is None and prev is not None and self.P.p_pair and self.pair_mem[prev] and self.rng.random() < self.P.p_pair:
                        succ = {w2: c for w2, c in self.pair_mem[prev].items() if w2 != prev}   # pair copying: continue a remembered word pair
                        if succ:                                             # (self-pairs are left to reduplication)
                            dup_word = self.rng.choices(list(succ), [c ** self.P.pair_gamma for c in succ.values()])[0]; self.stats["pair_copied"] += 1
                    for rnd in range(0 if dup_word else self.P.max_rounds):              # quirk 4: draw further rounds before giving up
                      for _ in range(self.P.K):
                        src, loc, how = self.choose_source(src_page, li_eff, pos, prev_src, history)
                        if self.slips[src] and self.rng.random() < self.P.p_slip_copy * sum(self.slips[src].values()) / self.n_written[src]:
                            forms = self.slips[src]; src = self.rng.choices(list(forms), list(forms.values()))[0]   # copy the slip as written
                            self.stats["slip_copied"] += 1
                        pool = history if len(history) < 50 else history[-400:]
                        pc = self.P.p_concat * (self.P.concat_edge_boost if cls in ("I", "F") else 1.0); r = self.rng.random(); route = None
                        if r < pc: w = self.op_concat(src, pool); route = "concat"
                        elif r < pc + self.P.p_multi: w = self.op_multi(src, pool); route = "multi"
                        elif how == "freq" and self.rng.random() < self.P.p_freq_verbatim: w = src      # a well-known word, written intact
                        else: w = src if how == "span" and self.rng.random() < 0.5 else self.mutate(src, pool)
                        if not w: continue
                        if self.flags["flow"]:
                            if prev is not None and self.rng.random() < self.P.p_flow:
                                w = self.flow_start(w, self._sample_target(self.T["bound"], last_g(prev), self.FIRST))
                            elif initial and self.rng.random() < self.P.p_flow_pos:
                                w = self.flow_start(w, self._sample_target(self.T["first"], "I", self.FIRST))
                            if final and self.rng.random() < self.P.p_flow_pos:
                                w = self.flow_end(w, self._sample_target(self.T["last"], "F", self.LAST))
                        override = None
                        if route == "concat":
                            override = 1.0 if len(glyphs(w)) <= self.P.concat_max else 0.0
                        elif self.P.bypass_w and w not in self.seen_written and self.flags["shape"]:
                            lp, illegal = self.T["shape"].logp(w)
                            if (illegal or lp < self.P.shape_floor) and 2 <= len(glyphs(w)) <= self.P.len_max and self.chunk_attested(w):
                                override = self.P.bypass_w; route = "bypass"
                        cands.append((w, loc if src_page is page else None)); routes.append(route)
                        wts.append(self.score(w, prev, cls, override))
                      if sum(wts) > 0: break
                      self.stats["extra_round"] += 1
                    if dup_word is not None:
                        w, loc = dup_word, None
                    elif sum(wts) > 0:
                        k = self.rng.choices(range(len(cands)), wts)[0]; w, loc = cands[k]
                        if routes[k] and w not in self.n_written:            # a new form from a one-off route
                            self.oneoff.add(w); self.stats["oneoff_" + routes[k]] += 1
                    else:                                                  # last resort: a recent legal word, verbatim
                        self.stats["verbatim_fallback"] += 1
                        w = next((h for h in reversed(history) if self.legal(h)), seed_line[0]); loc = None
                    if li == 0 and pos == 0:                               # page-initial gallows, glyph-aware, after selection
                        w = self.page_gallows(w)
                    # rare standalone-h slip: written on the page only; the intended form stays in the citation pool
                    ww = self.stray_h(w) if self.P.h_scale else w
                    written[li].append(ww); self.n_written[w] += 1
                    if ww != w: self.slips[w][ww] += 1
                    self.seen_written.add(ww)
                    page[li].append(w); history.append(w); page_cls[li].append(cls); self.sec_hist[self._sec].append(w)
                    if prev is not None: self.pair_mem[prev][w] += 1
                    self.sec_count[self._sec][w] += 1; self._n_total += 1
                    cls = self._cls
                    if prev is not None: self.trk_b.add(last_g(prev), first_g(w))
                    self.trk_f.add(cls, first_g(w)); self.trk_l.add(cls, last_g(w)); self.trk_n.add(cls, len_bin(w))
                    prev_src = loc
            out.append(written)
            prev_page = page[-3:]; prev_cls = page_cls[-3:]
        return out
