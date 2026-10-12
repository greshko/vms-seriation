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
sweep.py
========
Parameter sweep over synthetic foliation configurations.

For each combination of (source_text, words_per_folio, bifolios_per_quire,
distance_metric), this script:
  1. Generates a synthetic foliation via foliation.py
  2. Fits an NMF ensemble (K = 5–15, multiple seeds)
  3. Builds a weighted pairwise distance matrix
  4. Optimises a seriation order via SA + Or-opt + 2-opt
  5. Computes bifolio adjacency rate vs. random baseline
  6. Writes per-text and combined CSVs and a combined visualisation

All *.txt files found in TEXTS_DIR are processed. Results are written both
per-text (outputs/<stem>/results_full.csv etc.) and combined across all texts
(outputs/results_full_all.csv, outputs/results_summary_all.csv,
outputs/results_viz_all.html).

NOTE: Use for conducting seriation only. sweep_v2.py contains correct version of 
random baseline calcuation for calculating conjoint adjacency
ratio over baseline (CAROB).

Configuration
-------------
Edit the CONFIGURATION block below before running.

Usage
-----
    python sweep.py

Requires foliation.py in the same directory.
"""

from __future__ import annotations

import itertools
import json
import math
import random
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import randomized_svd

import foliation as fol

# =============================================================================
# CONFIGURATION
# =============================================================================

TEXTS_DIR    = Path("texts")       # all *.txt files here are processed
OUTPUT_DIR   = Path("outputs")

# Parameter grid
WORDS_PER_FOLIO_VALUES    = [200, 400, 600]
BIFOLIOS_PER_QUIRE_VALUES = [1, 2, 3, 4, 5]
DISTANCE_METRICS          = ["hellinger", "js", "l1"]

# NMF
K_MIN        = 5
K_MAX        = 15
NMF_SEEDS    = 5          # seeds per (K, combination)

# Seriation
N_RESTARTS   = 8          # independent SA runs per combination; best is kept
SA_MOVES     = ["2opt", "or1", "or2", "or3"]   # move types used in SA

# Reproducibility
GLOBAL_SEED  = 42

# =============================================================================
# CONSTANTS
# =============================================================================

_EPS = 1e-12
rng = np.random.default_rng(GLOBAL_SEED)

# =============================================================================
# DISTANCE FUNCTIONS
# =============================================================================

def _kl(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, _EPS, 1.0)
    q = np.clip(q, _EPS, 1.0)
    return float(np.sum(p * np.log(p / q)))

def _js(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    return float(math.sqrt(0.5 * _kl(p, m) + 0.5 * _kl(q, m)))

def _hellinger(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.linalg.norm(
        np.sqrt(np.clip(p, 0, 1)) - np.sqrt(np.clip(q, 0, 1))
    ) / math.sqrt(2.0))

def _l1(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.abs(p - q).sum() * 0.5)

DIST_FN = {
    "hellinger": _hellinger,
    "js":        _js,
    "l1":        _l1,
}

# =============================================================================
# NMF ENSEMBLE
# =============================================================================

def build_nmf_ensemble(
    counts: pd.DataFrame,
    k_min: int,
    k_max: int,
    n_seeds: int,
    base_seed: int = 0,
) -> tuple[dict, dict, dict]:
    """
    Fit NMF for K in [k_min, k_max] with n_seeds each.

    Returns
    -------
    doc_topic_by_KS : dict (K, s) -> ndarray (n_docs, K)
    errors_by_KS    : dict (K, s) -> float
    weights_by_KS   : dict (K, s) -> float  (inv-error normalised)
    """
    tfidf = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True)
    X = tfidf.fit_transform(counts.values)

    doc_topic_by_KS: dict = {}
    errors_by_KS:    dict = {}

    for i, K in enumerate(range(k_min, k_max + 1)):
        for s in range(n_seeds):
            nmf = NMF(
                n_components=K,
                init="nndsvda",
                random_state=base_seed + 1000 * i + s,
                max_iter=3000,
            )
            W = nmf.fit_transform(X)
            W = normalize(W, norm="l1", axis=1)
            doc_topic_by_KS[(K, s)] = W
            errors_by_KS[(K, s)]    = nmf.reconstruction_err_

    raw_w = np.array([1.0 / max(e, _EPS) for e in errors_by_KS.values()])
    raw_w /= raw_w.sum()
    ks_keys = list(errors_by_KS.keys())
    weights_by_KS = {ks: float(raw_w[j]) for j, ks in enumerate(ks_keys)}

    return doc_topic_by_KS, errors_by_KS, weights_by_KS


# =============================================================================
# DISTANCE MATRIX
# =============================================================================

def build_distance_matrix(
    doc_topic_by_KS: dict,
    weights_by_KS:   dict,
    metric:          str,
    n_docs:          int,
) -> np.ndarray:
    """Build weighted-average pairwise distance matrix (n_docs × n_docs)."""
    dist_fn = DIST_FN[metric]
    D = np.zeros((n_docs, n_docs), dtype=np.float32)

    for (K, s), w in weights_by_KS.items():
        W = doc_topic_by_KS[(K, s)]          # (n_docs, K)
        for a in range(n_docs):
            for b in range(a + 1, n_docs):
                d = dist_fn(W[a], W[b]) * w
                D[a, b] += d
                D[b, a] += d

    return D


# =============================================================================
# SERIATION: move generators
# =============================================================================

def _path_cost(order: list[int], D: np.ndarray) -> float:
    return float(sum(D[order[i], order[i + 1]] for i in range(len(order) - 1)))


def _apply_2opt(order: list[int], i: int, j: int) -> list[int]:
    """Reverse segment [i+1 .. j]."""
    return order[: i + 1] + order[i + 1 : j + 1][::-1] + order[j + 1 :]


def _apply_or_opt(order: list[int], i: int, length: int, j: int) -> list[int]:
    """
    Remove a chain of `length` nodes starting at position i,
    insert after position j (in the order with the chain removed).
    """
    chain = order[i : i + length]
    rest  = order[:i] + order[i + length :]
    # j is an index into `rest`; insert after rest[j]
    return rest[: j + 1] + chain + rest[j + 1 :]


# =============================================================================
# SERIATION: SA + Or-opt + 2-opt
# =============================================================================

def _auto_temperature(D: np.ndarray, n: int) -> tuple[float, float]:
    """
    Estimate initial and final SA temperatures from the distance matrix.

    T_init: set so that a move that worsens cost by one average edge weight
            is accepted with probability ~0.8.
    T_final: set to 1/1000 of T_init (effectively zero acceptance of bad moves).
    """
    # Sample a subset of edge weights to estimate scale
    sample_size = min(n * (n - 1) // 2, 500)
    idx_pairs = [
        (a, b)
        for a, b in zip(
            rng.integers(0, n, sample_size),
            rng.integers(0, n, sample_size),
        )
        if a != b
    ]
    if not idx_pairs:
        return 1.0, 0.001
    avg_edge = float(np.mean([D[a, b] for a, b in idx_pairs]))
    T_init  = -avg_edge / math.log(0.8)
    T_final = T_init / 1000.0
    return T_init, T_final


def _sa_seriate(
    D:          np.ndarray,
    init_order: list[int],
    seed:       int,
    n_steps_multiplier: int = 200,
) -> tuple[list[int], float]:
    """
    Simulated annealing seriation with 2-opt, Or-opt(1), Or-opt(2), Or-opt(3).

    Parameters
    ----------
    D            : pairwise distance matrix
    init_order   : starting permutation
    seed         : RNG seed for this restart
    n_steps_multiplier : total steps = n * multiplier

    Returns
    -------
    best_order, best_cost
    """
    local_rng = random.Random(seed)
    n = len(init_order)
    T_init, T_final = _auto_temperature(D, n)

    total_steps = n * n_steps_multiplier
    # Geometric cooling: T(t) = T_init * (T_final/T_init)^(t/total_steps)
    cooling_ratio = (T_final / T_init) ** (1.0 / max(total_steps, 1))

    current  = init_order[:]
    cur_cost = _path_cost(current, D)
    best     = current[:]
    best_cost = cur_cost

    T = T_init

    for step in range(total_steps):
        move_type = local_rng.choice(SA_MOVES)

        if move_type == "2opt":
            i = local_rng.randint(0, n - 3)
            j = local_rng.randint(i + 2, n - 1)
            candidate = _apply_2opt(current, i, j)

        else:  # or-opt with chain length 1, 2, or 3
            length = int(move_type[2])   # "or1" -> 1, "or2" -> 2, "or3" -> 3
            if n < length + 2:
                T *= cooling_ratio
                continue
            i = local_rng.randint(0, n - length - 1)
            # j indexes into the remaining sequence after removing the chain
            j = local_rng.randint(0, n - length - 2)
            # Avoid inserting back into the same position (no-op)
            if j >= i:
                j += 1
            if j > n - length - 1:
                T *= cooling_ratio
                continue
            candidate = _apply_or_opt(current, i, length, j)

        cand_cost = _path_cost(candidate, D)
        delta = cand_cost - cur_cost

        if delta < 0 or local_rng.random() < math.exp(-delta / max(T, _EPS)):
            current  = candidate
            cur_cost = cand_cost
            if cur_cost < best_cost:
                best      = current[:]
                best_cost = cur_cost

        T *= cooling_ratio

    return best, best_cost


# =============================================================================
# INITIAL ORDER: 1D SVD projection
# =============================================================================

def _svd_initial_order(
    doc_topic_by_KS: dict,
    weights_by_KS:   dict,
    n_docs:          int,
    max_K:           int,
) -> list[int]:
    """Project weighted average topic vectors to 1D via SVD; sort."""
    avg_emb = np.zeros((n_docs, max_K))
    for (K, s), w in weights_by_KS.items():
        W = doc_topic_by_KS[(K, s)]
        padded = np.zeros((n_docs, max_K))
        padded[:, :K] = W
        avg_emb += w * padded

    U, svals, _ = randomized_svd(avg_emb, n_components=1, random_state=GLOBAL_SEED)
    one_d = (U[:, 0] * svals[0]).tolist()
    return sorted(range(n_docs), key=lambda j: one_d[j])


# =============================================================================
# ADJACENCY SCORING
# =============================================================================

def bifolio_adjacency_rate(
    order:        list[int],
    doc_ids:      list[int],
    folio_bifolio: dict[int, int],
) -> float:
    """
    Fraction of consecutive pairs in `order` that share a bifolio_global.

    Parameters
    ----------
    order         : list of row indices (0-based) into doc_ids
    doc_ids       : list mapping row index -> doc_id (folio number)
    folio_bifolio : dict mapping doc_id -> bifolio_global
    """
    n = len(order)
    if n < 2:
        return 0.0
    hits = sum(
        1
        for i in range(n - 1)
        if folio_bifolio[doc_ids[order[i]]] == folio_bifolio[doc_ids[order[i + 1]]]
    )
    return hits / (n - 1)


# =============================================================================
# VOCABULARY PRUNING
# =============================================================================

def prune_vocabulary(counts: pd.DataFrame, n_max: int = 3000) -> pd.DataFrame:
    """Remove tokens in >80% of docs or <2 docs; keep top n_max by frequency."""
    n_docs   = counts.shape[0]
    doc_freq = (counts > 0).sum(axis=0)
    mask     = (doc_freq >= 2) & (doc_freq <= 0.8 * n_docs)
    pruned   = counts.loc[:, mask]
    if pruned.shape[1] > n_max:
        top = counts.sum(axis=0)[mask].sort_values(ascending=False).head(n_max).index
        pruned = pruned[top]
    return pruned


# =============================================================================
# SINGLE COMBINATION RUNNER
# =============================================================================

def run_combination(
    text_path: Path,
    wpf:     int,
    bpq:     int,
    metric:  str,
    combo_idx: int,
    n_combos:  int,
) -> list[dict]:
    """
    Run the full pipeline for one (text, words_per_folio, bifolios_per_quire, metric).

    Returns a list of result dicts, one per restart.
    """
    t0 = time.time()
    source = text_path.stem
    tag = f"{source} | wpf={wpf} bpq={bpq} metric={metric}"
    print(f"  [{combo_idx}/{n_combos}] {tag} ... ", end="", flush=True)

    # --- 1. Foliation ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        df = fol.generate_foliation(text_path, wpf, bpq)

    summary = fol.foliation_summary(df)
    n_folios = summary["n_folios"]

    baseline = fol.random_baseline(df)

    # One row per folio for doc_id -> bifolio_global mapping
    folio_meta = (
        df[["doc_id", "bifolio_global"]]
        .drop_duplicates("doc_id")
        .set_index("doc_id")["bifolio_global"]
        .to_dict()
    )

    # --- 2. Count matrix ---
    counts_full = (
        df.groupby(["doc_id", "token"])
          .size()
          .unstack(fill_value=0)
          .sort_index()
    )
    counts = prune_vocabulary(counts_full)

    if counts.shape[1] == 0:
        print(f"SKIP (no vocabulary after pruning)")
        return []

    doc_ids = counts.index.tolist()   # list of folio numbers (int)
    n_docs  = len(doc_ids)

    if n_docs < 2:
        print(f"SKIP (only {n_docs} folio(s))")
        return []

    # --- 3. NMF ensemble ---
    base_seed = abs(hash((wpf, bpq, metric))) % (2**31)
    doc_topic_by_KS, errors_by_KS, weights_by_KS = build_nmf_ensemble(
        counts, K_MIN, K_MAX, NMF_SEEDS, base_seed=base_seed
    )

    # --- 4. Distance matrix ---
    D = build_distance_matrix(doc_topic_by_KS, weights_by_KS, metric, n_docs)

    # --- 5. Seriation: multiple restarts ---
    svd_order = _svd_initial_order(doc_topic_by_KS, weights_by_KS, n_docs, K_MAX)

    results = []
    for restart in range(N_RESTARTS):
        # Vary start: SVD order for restart 0, random permutations otherwise
        if restart == 0:
            init = svd_order[:]
        else:
            init = list(rng.permutation(n_docs))

        order, cost = _sa_seriate(
            D, init,
            seed=base_seed + restart,
            n_steps_multiplier=200,
        )

        adj_rate = bifolio_adjacency_rate(order, doc_ids, folio_meta)

        results.append({
            "source_text":        source,
            "words_per_folio":    wpf,
            "bifolios_per_quire": bpq,
            "metric":             metric,
            "restart":            restart,
            "n_folios":           n_docs,
            "n_quires":           summary["n_quires"],
            "n_bifolios":         summary["n_bifolios"],
            "bifolio_adj_rate":   round(adj_rate, 6),
            "random_baseline":    round(baseline, 6),
            "lift_over_baseline": round(adj_rate - baseline, 6),
            "seriation_cost":     round(float(cost), 6),
        })

    elapsed = time.time() - t0
    best_rate = max(r["bifolio_adj_rate"] for r in results)
    print(f"done in {elapsed:.1f}s  best_adj={best_rate:.4f}  baseline={baseline:.4f}")

    return results


# =============================================================================
# VISUALISATION (self-contained HTML)
# =============================================================================

def build_viz_html(summary_df: pd.DataFrame, title: str = "Seriation Sweep") -> str:
    """
    Build a self-contained HTML file with an interactive chart.
    The chart shows mean bifolio adjacency rate vs bifolios_per_quire,
    faceted by words_per_folio, with one line per distance metric.
    The full summary data is embedded as JSON.
    """
    data_json = summary_df.to_json(orient="records")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Bifolio Adjacency Rate — {title}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  body {{
    font-family: system-ui, sans-serif;
    max-width: 1100px;
    margin: 2rem auto;
    padding: 0 1rem;
    background: #fafafa;
    color: #222;
  }}
  h1 {{ font-size: 1.4rem; margin-bottom: 0.25rem; }}
  p.sub {{ color: #555; font-size: 0.9rem; margin-top: 0; }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: 1.5rem;
    margin-top: 1.5rem;
  }}
  .card {{
    background: #fff;
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 1rem;
  }}
  canvas {{ width: 100% !important; }}
  .legend {{
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin-bottom: 0.5rem;
    font-size: 0.85rem;
  }}
  .legend-item {{ display: flex; align-items: center; gap: 0.3rem; }}
  .legend-dot {{ width: 12px; height: 12px; border-radius: 50%; display: inline-block; }}
</style>
</head>
<body>
<h1>Bifolio Adjacency Rate — {title}</h1>
<p class="sub">
  Mean adjacency rate (± 1 SD across restarts) vs bifolios per quire,
  by distance metric. Dashed line = random baseline.
</p>

<div class="legend">
  <span class="legend-item"><span class="legend-dot" style="background:#2563eb"></span>Hellinger</span>
  <span class="legend-item"><span class="legend-dot" style="background:#16a34a"></span>JS divergence</span>
  <span class="legend-item"><span class="legend-dot" style="background:#dc2626"></span>L1</span>
  <span class="legend-item"><span class="legend-dot" style="background:#aaa; opacity:0.6"></span>Random baseline</span>
</div>

<div class="grid" id="charts"></div>

<script>
const RAW = {data_json};

const METRICS = ["hellinger", "js", "l1"];
const COLORS  = {{ hellinger: "#2563eb", js: "#16a34a", l1: "#dc2626" }};
const WPF_VALUES = [...new Set(RAW.map(d => d.words_per_folio))].sort((a,b) => a-b);
const BPQ_VALUES = [...new Set(RAW.map(d => d.bifolios_per_quire))].sort((a,b) => a-b);

function getRows(wpf, metric) {{
  return BPQ_VALUES.map(bpq => {{
    const rows = RAW.filter(d =>
      d.words_per_folio === wpf &&
      d.bifolios_per_quire === bpq &&
      d.metric === metric
    );
    if (!rows.length) return {{ mean: null, sd: null, baseline: null }};
    const vals = rows.map(r => r.bifolio_adj_rate_mean);
    const sds  = rows.map(r => r.bifolio_adj_rate_sd);
    return {{
      mean:     vals[0],
      sd:       sds[0],
      baseline: rows[0].random_baseline_mean,
    }};
  }});
}}

const container = document.getElementById("charts");

WPF_VALUES.forEach(wpf => {{
  const card = document.createElement("div");
  card.className = "card";
  card.innerHTML = `<h3 style="margin:0 0 0.75rem;font-size:1rem">
    ${{wpf}} words per folio</h3><canvas id="chart-${{wpf}}"></canvas>`;
  container.appendChild(card);

  const datasets = [];

  // Baseline (same for all metrics at given wpf, bpq)
  const baselineVals = BPQ_VALUES.map(bpq => {{
    const row = RAW.find(d =>
      d.words_per_folio === wpf && d.bifolios_per_quire === bpq
    );
    return row ? row.random_baseline_mean : null;
  }});

  datasets.push({{
    label: "Random baseline",
    data: baselineVals,
    borderColor: "#bbb",
    borderDash: [5, 4],
    borderWidth: 1.5,
    pointRadius: 0,
    fill: false,
    tension: 0.3,
  }});

  METRICS.forEach(metric => {{
    const rows = getRows(wpf, metric);
    datasets.push({{
      label: metric,
      data: rows.map(r => r.mean),
      borderColor: COLORS[metric],
      backgroundColor: COLORS[metric] + "22",
      borderWidth: 2,
      pointRadius: 4,
      fill: false,
      tension: 0.3,
    }});
  }});

  new Chart(document.getElementById(`chart-${{wpf}}`), {{
    type: "line",
    data: {{ labels: BPQ_VALUES.map(String), datasets }},
    options: {{
      responsive: true,
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{ mode: "index", intersect: false }},
      }},
      scales: {{
        x: {{
          title: {{ display: true, text: "Bifolios per quire" }},
        }},
        y: {{
          title: {{ display: true, text: "Bifolio adjacency rate" }},
          min: 0,
          suggestedMax: 0.3,
        }},
      }},
    }},
  }});
}});
</script>
</body>
</html>
"""
    return html


# =============================================================================
# MAIN
# =============================================================================

def _summarise(full_df: pd.DataFrame, groupby_cols: list[str]) -> pd.DataFrame:
    """Aggregate a full results dataframe into a summary."""
    return (
        full_df
        .groupby(groupby_cols)
        .agg(
            bifolio_adj_rate_mean   = ("bifolio_adj_rate",   "mean"),
            bifolio_adj_rate_sd     = ("bifolio_adj_rate",   "std"),
            bifolio_adj_rate_max    = ("bifolio_adj_rate",   "max"),
            random_baseline_mean    = ("random_baseline",    "mean"),
            lift_over_baseline_mean = ("lift_over_baseline", "mean"),
            seriation_cost_mean     = ("seriation_cost",     "mean"),
            n_folios                = ("n_folios",           "first"),
            n_quires                = ("n_quires",           "first"),
            n_bifolios              = ("n_bifolios",         "first"),
            n_restarts              = ("restart",            "count"),
        )
        .reset_index()
        .round(6)
    )


def _console_summary(full_df: pd.DataFrame, label: str) -> None:
    """Print a brief bpq × adjacency_rate table to stdout."""
    print(f"\n--- {label}: mean adjacency rate by bpq ---")
    tbl = (
        full_df
        .groupby("bifolios_per_quire")["bifolio_adj_rate"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "adj_mean", "std": "adj_sd", "count": "n"})
        .round(4)
    )
    print(tbl.to_string())
    print(f"\n--- {label}: lift over baseline by bpq ---")
    lift = (
        full_df
        .groupby("bifolios_per_quire")["lift_over_baseline"]
        .mean()
        .round(4)
    )
    print(lift.to_string())


def main() -> None:
    # ------------------------------------------------------------------
    # Discover input texts
    # ------------------------------------------------------------------
    text_files = sorted(TEXTS_DIR.glob("*.txt"))
    if not text_files:
        print(f"No *.txt files found in {TEXTS_DIR}. Exiting.")
        return

    print("=" * 60)
    print("Seriation sweep — bifolio adjacency rate")
    print(f"  Texts dir : {TEXTS_DIR}  ({len(text_files)} file(s) found)")
    for tf in text_files:
        print(f"              {tf.name}")
    print(f"  Grid      : {len(WORDS_PER_FOLIO_VALUES)} wpf × "
          f"{len(BIFOLIOS_PER_QUIRE_VALUES)} bpq × "
          f"{len(DISTANCE_METRICS)} metrics = "
          f"{len(WORDS_PER_FOLIO_VALUES)*len(BIFOLIOS_PER_QUIRE_VALUES)*len(DISTANCE_METRICS)} combos/text")
    print(f"  NMF       : K={K_MIN}–{K_MAX}, {NMF_SEEDS} seeds")
    print(f"  SA        : {N_RESTARTS} restarts")
    print("=" * 60)

    combos = list(itertools.product(
        WORDS_PER_FOLIO_VALUES,
        BIFOLIOS_PER_QUIRE_VALUES,
        DISTANCE_METRICS,
    ))
    n_combos_per_text = len(combos)

    all_results: list[dict] = []   # accumulates across all texts

    # ------------------------------------------------------------------
    # Per-text sweep
    # ------------------------------------------------------------------
    for text_path in text_files:
        stem = text_path.stem
        print(f"\n{'─' * 60}")
        print(f"  Text: {stem}")
        print(f"{'─' * 60}")

        # Per-text output directory
        text_out_dir = OUTPUT_DIR / stem
        text_out_dir.mkdir(parents=True, exist_ok=True)

        text_results: list[dict] = []

        for idx, (wpf, bpq, metric) in enumerate(combos, start=1):
            results = run_combination(
                text_path, wpf, bpq, metric,
                idx, n_combos_per_text,
            )
            text_results.extend(results)

        if not text_results:
            print(f"  No results for {stem}. Skipping.")
            continue

        text_full_df = pd.DataFrame(text_results)
        all_results.extend(text_results)

        # Per-text full CSV
        per_full_path = text_out_dir / "results_full.csv"
        text_full_df.to_csv(per_full_path, index=False)
        print(f"\n  Full results  -> {per_full_path}  ({len(text_full_df)} rows)")

        # Per-text summary CSV
        per_summary_df = _summarise(
            text_full_df,
            ["words_per_folio", "bifolios_per_quire", "metric"],
        )
        per_summary_path = text_out_dir / "results_summary.csv"
        per_summary_df.to_csv(per_summary_path, index=False)
        print(f"  Summary       -> {per_summary_path}  ({len(per_summary_df)} rows)")

        # Per-text visualisation
        per_viz_html = build_viz_html(per_summary_df, title=stem)
        per_viz_path = text_out_dir / "results_viz.html"
        per_viz_path.write_text(per_viz_html, encoding="utf-8")
        print(f"  Visualisation -> {per_viz_path}")

        _console_summary(text_full_df, stem)

    # ------------------------------------------------------------------
    # Combined outputs (all texts together)
    # ------------------------------------------------------------------
    if not all_results:
        print("\nNo results generated across any text. Check TEXTS_DIR.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_df = pd.DataFrame(all_results)

    # Combined full CSV
    all_full_path = OUTPUT_DIR / "results_full_all.csv"
    all_df.to_csv(all_full_path, index=False)
    print(f"\n{'=' * 60}")
    print(f"Combined full    -> {all_full_path}  ({len(all_df)} rows)")

    # Combined summary CSV — group by source_text too
    all_summary_df = _summarise(
        all_df,
        ["source_text", "words_per_folio", "bifolios_per_quire", "metric"],
    )
    all_summary_path = OUTPUT_DIR / "results_summary_all.csv"
    all_summary_df.to_csv(all_summary_path, index=False)
    print(f"Combined summary -> {all_summary_path}  ({len(all_summary_df)} rows)")

    # Combined visualisation: average over source texts so the chart
    # shape is directly comparable across texts on one page.
    # We pass the summary averaged over source_text for the chart data,
    # but the full per-source data is in the CSV for deeper analysis.
    combined_viz_summary = _summarise(
        all_df,
        ["words_per_folio", "bifolios_per_quire", "metric"],
    )
    all_viz_html = build_viz_html(
        combined_viz_summary,
        title=f"All texts combined ({len(text_files)} source(s))",
    )
    all_viz_path = OUTPUT_DIR / "results_viz_all.html"
    all_viz_path.write_text(all_viz_html, encoding="utf-8")
    print(f"Combined viz     -> {all_viz_path}")

    _console_summary(all_df, "ALL TEXTS")
    print("\nDone.")


if __name__ == "__main__":
    main()
