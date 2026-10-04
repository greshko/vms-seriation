"""
boxplot_v2.py
=============
Box-and-whisker plot of CAROB (conjoint adjacency ratio over baseline) per quire
construction for ONE text -- a companion to violin_v2.py's chart.

violin_v2 compares many texts (one median per text per quire construction) and
tests them statistically. This script does the opposite on purpose: it takes a
single text, applies no inference, and shows in detail how that text's CAROB
moves across the runs behind each quire construction. The VMS is drawn exactly as
in violin_v2.py: a dashed median line inside nested percentile bands of the VMS
run values, the CAROB = 1 reference line, the same colors, axis rules, legend
position and off-scale marker, so the two charts can sit side by side.

What each box shows
-------------------
One box per quire construction (Singulion ... Quinion; groups come from
bifolios_per_quire). Every run of the text in that group is one value: each
(window x words_per_folio x distance metric x SA restart) row of sweep_v2's
results. Box = 25th-75th percentile, line = median, whiskers = 5th and 95th
percentiles (the same central 50% / central 90% used for the VMS bands). Add
--points to also draw every individual run as a dot (off by default: the dots are
hard to read at small print sizes; coloured by words per folio when there are
several sizes, else by metric). Percentiles are configurable (--box, --whiskers);
--whiskers 0 100 makes the whiskers span the full min-max range.

Reading it. The spread is run-to-run variability of ONE document -- distance
metric, optimiser restarts, folio size (here a +/-8-word neighbourhood, i.e. boundary
placement) and (if the text is longer than the window) bootstrap windows -- not
text-to-text variability, so it is not comparable to the violins' spread. The VMS bands
contain only metric x restart spread. CAROB scales with the number of folios, so
pooling widely different folio sizes would widen the boxes; the default neighbourhood
changes the folio count by only a few percent. (With --points, dots are coloured by
words per folio when there are several, otherwise by metric.)

Two ways to run
---------------
1. From the text (runs sweep_v2.py, then plots):
       python boxplot_v2.py merged_voynich_like.txt --vms-input vms_results.csv
   Default settings: a NEIGHBOURHOOD of five folio sizes around the VMS's average of
   378 words per folio (38,525 tokens / 102 folios = 377.70, measured from
   --vms-foliation): 370, 374, 378, 382, 386; and 10 SA restarts per metric at each
   (5 sizes x 3 metrics x 10 restarts = 150 runs per quire construction).
   Why a neighbourhood and not just 378: for one text, CAROB is jagged in the exact
   folio size (it depends on where the folio boundaries fall relative to the text's
   content), and more restarts only average optimiser noise, not that effect. Pooling
   nearby sizes averages over boundary placement while staying at the VMS's scale.
   Override with --wpf (e.g. --wpf 378 --restarts 50 for a single size, or
   --wpf 200 400 600) and --restarts. Restarts are optimiser repeats on the same
   text, not independent samples of it.
   Window length = the VMS token count (from --vms-foliation) when the text is at
   least that long (then --windows bootstrap windows are drawn); if the text is
   shorter, the whole text is used as one window and a note says so.
2. From existing results (no sweep):
       python boxplot_v2.py --results outputs_v2/mytext/results_full.csv
       python boxplot_v2.py --results outputs_v2/results_full_all.csv --source-text mytext

Tail report (--tail-report; needs --vms-input)
----------------------------------------------
Descriptive, not a hypothesis test. For each quire construction it reports, from this
text's runs: P(run >= VMS median) -- how often one run of this text reaches the VMS's
CAROB -- and the share of runs inside each VMS percentile band, with 95% intervals
that respect how the runs were produced. The runs are NOT independent: those from the
same (folio size x metric) cell share a distance matrix and converge to similar
orderings, so the intervals resample CELLS first and restarts within them (a two-stage
bootstrap). Reported intervals are the wider of that and the naive Wilson interval,
because the bootstrap degenerates at 0 or all (e.g. "[0, 0]" when no run reaches the
VMS). A second interval also resamples the VMS's own runs (metric cells, then
restarts), since the VMS median moves with the metric. Also written: the intraclass
correlation of the cells and the effective number of independent runs. Wilcoxon-type
p-values are deliberately not offered: n here is set by the restart budget, runs are
clustered, and there is no population of texts to generalise to. Read the numbers as
statements about THIS text's run-to-run variability, not about the VMS's origin or
about other texts.

Outputs (in --out-dir, default ./boxplot_v2_out)
-------
  boxplot_single_text.svg    the chart (add --png for a PNG too)
  boxplot_percentiles.csv    per quire construction: n runs, min, percentiles, max
  boxplot_breakdown.csv      per quire construction x words/folio x metric: n, median,
                             min, max (the detail behind the spread)
  vms_percentiles.csv        VMS run summary behind the bands (as violin_v2 writes it)
  tail_report.csv            (--tail-report) per quire construction: P(run >= VMS median),
                             share of runs inside the VMS bands, intervals, ICC, effective n
  run_settings.json
  sweep/                     sweep_v2 output (text mode only)

Matching the violin chart. Use the same --wpf / --metrics / --best-cost-only /
--vms-bands and the same --ymax on both charts. --metrics and --best-cost-only
apply to the VMS rows too; --wpf applies to the text only (the VMS has no
words-per-folio value).

Requires violin_v2.py (helpers), foliation.py and sweep_v2.py (text mode) in the
same directory.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import violin_v2 as v            # shared data loading, filters, VMS bands, group labels

SWEEP_V2 = Path(__file__).resolve().parent / "sweep_v2.py"
VMS_FOLIATION_CSV = Path("voynich/vms_rf1b_folio_bifolio.csv")
N_WINDOWS_DEFAULT = 10
RESTARTS_DEFAULT = 10          # SA restarts per metric per folio size in text mode (sweep_v2: 8)
WPF_FALLBACK = 378             # VMS average words per folio (38,525 / 102 = 377.70), used only
                               # if the VMS foliation CSV cannot be read
WPF_OFFSETS = (-8, -4, 0, 4, 8)   # default folio sizes = VMS average (rounded) + these
BOX_DEFAULT = (25.0, 75.0)
WHISKERS_DEFAULT = (5.0, 95.0)
VMS_COLOR = "#e05c2a"
_WPF_COLORS = ["#2b6cb0", "#2f855a", "#805ad5", "#d53f8c", "#718096"]   # no orange/brown: orange is reserved for VMS
_METRIC_COLORS = {"hellinger": "#2563eb", "js": "#16a34a", "l1": "#dc2626"}


# =============================================================================
# TEXT MODE: run sweep_v2 on the text
# =============================================================================

def vms_token_count(csv_path: Path) -> int:
    """Tokens in the VMS foliation CSV (read verbatim; empty tokens dropped)."""
    tok = pd.read_csv(csv_path, encoding="utf-8-sig", keep_default_na=False,
                      na_values=[""], dtype={"token": str}, usecols=["token"])["token"]
    return int(tok.notna().sum())


def vms_avg_words_per_folio(csv_path: Path) -> tuple[float, int, int]:
    """(average tokens per folio, total tokens, number of folios) of the VMS foliation CSV,
    folios = distinct doc_id (the same unit voynich_run_v2 and sweep_v3 use)."""
    raw = pd.read_csv(csv_path, encoding="utf-8-sig", keep_default_na=False, na_values=[""],
                      dtype={"token": str}, usecols=["token", "doc_id"]).dropna(subset=["token"])
    n_tok, n_fol = len(raw), int(raw["doc_id"].nunique())
    return n_tok / n_fol, n_tok, n_fol


def run_sweep(text: Path, out: Path, window_len: int, windows: int, args) -> Path:
    """Run sweep_v2.py on one text; return the path of its results_full.csv."""
    cmd = [sys.executable, str(SWEEP_V2), str(text), "--window-len", str(window_len),
           "--windows", str(windows), "--out", str(out)]
    for flag, val in (("--wpf", args.wpf), ("--bpq", args.bpq), ("--metrics", args.metrics)):
        if val:
            cmd += [flag] + [str(x) for x in val]
    for flag, val in (("--restarts", args.restarts), ("--nmf-seeds", args.nmf_seeds),
                      ("--sa-steps", args.sa_steps), ("--workers", args.workers),
                      ("--seed", args.seed)):
        if val is not None:
            cmd += [flag, str(val)]
    if args.force:
        cmd.append("--force")
    print("\n$ " + " ".join(cmd), flush=True)
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        raise SystemExit(f"sweep_v2.py failed (exit code {rc}).")
    res = out / text.stem / "results_full.csv"
    if not res.exists():
        raise SystemExit(f"sweep_v2 produced no results for {text.name} (expected {res}).")
    return res


# =============================================================================
# SUMMARIES
# =============================================================================

def percentile_table(data: dict[str, np.ndarray], groups: list[str], box, whis) -> pd.DataFrame:
    rows = []
    for g in groups:
        x = data[g]
        q = np.percentile(x, [whis[0], box[0], 50, box[1], whis[1]])
        rows.append({"group": g, "n_runs": len(x), "min": x.min(),
                     f"p{whis[0]:g}": q[0], f"p{box[0]:g}": q[1], "median": q[2],
                     f"p{box[1]:g}": q[3], f"p{whis[1]:g}": q[4], "max": x.max()})
    return pd.DataFrame(rows)


def breakdown_table(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["bpq_label"] + [c for c in ("words_per_folio", "metric") if c in df.columns]
    out = (df.groupby(keys)[v.VALUE_COL]
             .agg(n_runs="size", median="median", min="min", max="max").reset_index()
             .rename(columns={"bpq_label": "group"}))
    return out.round(4)


# =============================================================================
# TAIL REPORT (descriptive; see the module docstring)
# =============================================================================

_CELL_KEYS_TEXT = ("window_idx", "words_per_folio", "metric")   # a cell = one distance matrix
_CELL_KEYS_VMS = ("window_idx", "metric")


def _wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson 95% interval for a proportion k/n (naive: assumes independent runs)."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def _cells(df: pd.DataFrame, keys) -> list[np.ndarray]:
    keys = [k for k in keys if k in df.columns]
    if not keys:                                   # no structure known: one cell
        return [df[v.VALUE_COL].to_numpy(dtype=float)]
    return [g[v.VALUE_COL].to_numpy(dtype=float) for _, g in df.groupby(keys)]


def _two_stage(cells: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """Cluster bootstrap resample: draw cells with replacement, then draw runs with
    replacement within each drawn cell."""
    pick = rng.integers(0, len(cells), len(cells))
    return np.concatenate([cells[j][rng.integers(0, len(cells[j]), len(cells[j]))] for j in pick])


def _icc(cells: list[np.ndarray]) -> tuple[float, float]:
    """(intraclass correlation across cells, effective number of independent runs)."""
    k = len(cells)
    sizes = np.array([len(c) for c in cells], dtype=float)
    n = float(sizes.sum())
    if k < 2 or n <= k:
        return float("nan"), n
    gm = np.concatenate(cells).mean()
    msb = sum(len(c) * (c.mean() - gm) ** 2 for c in cells) / (k - 1)
    msw = sum(((c - c.mean()) ** 2).sum() for c in cells) / (n - k)
    n0 = (n - (sizes ** 2).sum() / n) / (k - 1)
    denom = msb + (n0 - 1) * msw
    icc = max(0.0, (msb - msw) / denom) if denom > 0 else 0.0
    return icc, n / (1 + (n / k - 1) * icc)


def _union(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return min(a[0], b[0]), max(a[1], b[1])


def tail_report(df: pd.DataFrame, groups: list[str], vms_df: pd.DataFrame, vms_value: float,
                coverages: list[float], n_boot: int, seed: int) -> pd.DataFrame:
    """One row per quire construction (see the module docstring)."""
    rng = np.random.default_rng(seed)
    vals = vms_df[v.VALUE_COL].to_numpy(dtype=float)
    bands = v.vms_band_table(vals, coverages)                 # [(coverage %, lo, hi), ...]
    vcells = _cells(vms_df, _CELL_KEYS_VMS)
    vms_meds = np.array([np.median(_two_stage(vcells, rng)) for _ in range(n_boot)])

    rows = []
    for g in groups:
        cells = _cells(df[df["bpq_label"] == g], _CELL_KEYS_TEXT)
        runs = np.concatenate(cells)
        n, k = len(runs), int((runs >= vms_value).sum())
        icc, n_eff = _icc(cells)
        ex_fixed, ex_noisy = np.empty(n_boot), np.empty(n_boot)
        inside = {c: np.empty(n_boot) for c, _, _ in bands}
        for b in range(n_boot):
            z = _two_stage(cells, rng)
            ex_fixed[b] = np.mean(z >= vms_value)
            ex_noisy[b] = np.mean(z >= vms_meds[b])
            for c, lo, hi in bands:
                inside[c][b] = np.mean((z >= lo) & (z <= hi))
        naive = _wilson(k, n)
        clus = tuple(np.percentile(ex_fixed, [2.5, 97.5]))
        rep_ci = _union(naive, clus)
        noisy = _union(tuple(np.percentile(ex_noisy, [2.5, 97.5])), rep_ci)
        row = {"group": g, "n_runs": n, "n_cells": len(cells), "icc": icc, "effective_n": n_eff,
               "vms_median": vms_value, "vms_runs": len(vals),
               "p_run_ge_vms_median": k / n,
               "ci_naive_lo": naive[0], "ci_naive_hi": naive[1],
               "ci_cluster_lo": clus[0], "ci_cluster_hi": clus[1],
               "ci_reported_lo": rep_ci[0], "ci_reported_hi": rep_ci[1],
               "ci_with_vms_noise_lo": noisy[0], "ci_with_vms_noise_hi": noisy[1],
               "note": ("none of the runs reaches the VMS median; upper end is the naive bound" if k == 0
                        else "every run reaches the VMS median; lower end is the naive bound" if k == n
                        else "")}
        for c, lo, hi in bands:
            share = float(np.mean((runs >= lo) & (runs <= hi)))
            ci = _union(_wilson(int(((runs >= lo) & (runs <= hi)).sum()), n),
                        tuple(np.percentile(inside[c], [2.5, 97.5])))
            row.update({f"in_vms{c:g}_share": share, f"in_vms{c:g}_ci_lo": ci[0],
                        f"in_vms{c:g}_ci_hi": ci[1]})
        rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs["vms_cells"] = len(vcells)
    return out


def print_tail_report(tab: pd.DataFrame, coverages: list[float]) -> None:
    covs = sorted(set(coverages))
    print("\n" + "=" * 78)
    print("Tail report: how often does a run of THIS text reach the VMS?  (descriptive; no test)")
    print("=" * 78)
    head = f"{'':11s} {'runs':>4s} {'eff.n':>5s} {'P(>=VMS med)':>12s} {'95% CI':>13s} {'incl.VMS noise':>15s}"
    for c in covs:
        head += f" {f'in VMS {c:g}% band (95% CI)':>29s}"
    print(head)
    for r in tab.itertuples(index=False):
        d = r._asdict()
        line = (f"{d['group']:11s} {d['n_runs']:4d} {d['effective_n']:5.0f} "
                f"{d['p_run_ge_vms_median']:12.3f} "
                f"{'[%.2f, %.2f]' % (d['ci_reported_lo'], d['ci_reported_hi']):>13s} "
                f"{'[%.2f, %.2f]' % (d['ci_with_vms_noise_lo'], d['ci_with_vms_noise_hi']):>15s}")
        for c in covs:
            line += f" {d[f'in_vms{c:g}_share']:7.2f} [{d[f'in_vms{c:g}_ci_lo']:.2f}, {d[f'in_vms{c:g}_ci_hi']:.2f}]".rjust(30)
        print(line)
    print("  eff.n = runs' worth of independent information (cells share distance matrices).")
    print("  95% CI = wider of a two-stage cluster bootstrap (cells, then restarts) and the naive "
          "Wilson interval.")
    nv = tab.attrs.get("vms_cells")
    print("  'incl. VMS noise' also resamples the VMS's own runs (its median moves with the metric)"
          + (f"; it rests on only {nv} VMS cell(s), so treat it as a rough indication." if nv else "."))
    if (tab["note"] != "").any():
        print("  Groups with 0 or all runs reaching the VMS: the bootstrap degenerates there, so "
              "the naive bound is used.")


# =============================================================================
# PLOT (mirrors violin_v2.make_violin)
# =============================================================================

def make_boxplot(data: dict[str, np.ndarray], cats: dict[str, np.ndarray] | None,
                 cat_colors: dict | None, cat_title: str | None, groups: list[str],
                 vms_value: float | None, out_path: Path, *, box, whis,
                 ymax: float | None = None, vms_n_runs: int | None = None,
                 vms_bands: list[tuple[float, float, float]] | None = None,
                 show_points: bool = False, png: bool = False) -> None:
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(9, 6))                       # same size as the violin chart
    cmap = matplotlib.colormaps["viridis"]
    colors = [cmap(p) for p in np.linspace(0.15, 0.85, len(groups))]
    pos = list(range(1, len(groups) + 1))

    stats = []
    for g in groups:
        q = np.percentile(data[g], [whis[0], box[0], 50, box[1], whis[1]])
        stats.append(dict(whislo=q[0], q1=q[1], med=q[2], q3=q[3], whishi=q[4], fliers=[]))
    bp = ax.bxp(stats, positions=pos, widths=0.55, patch_artist=True, showfliers=False,
                boxprops=dict(linewidth=1.2, edgecolor="#333"),
                medianprops=dict(color="#111", linewidth=2.2),
                whiskerprops=dict(color="#333", linewidth=1.2),
                capprops=dict(color="#333", linewidth=1.2))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(matplotlib.colors.to_rgba(c, 0.75))
        patch.set_zorder(1.6)
    for k in ("whiskers", "caps"):
        for ln in bp[k]:
            ln.set_zorder(1.7)
    for ln in bp["medians"]:
        ln.set_zorder(1.9)

    if show_points:
        rng = np.random.default_rng(0)
        for i, g in enumerate(groups):
            vals = data[g]
            c = ([cat_colors[x] for x in cats[g]] if cats is not None else colors[i])
            ax.scatter(i + 1 + rng.uniform(-0.14, 0.14, size=len(vals)), vals, color=c,
                       alpha=0.65, s=20, zorder=3, edgecolors="white", linewidths=0.4)

    # Explicit y-limits (as in violin_v2): unless --ymax is given, the axis always
    # extends far enough to include the VMS value and its outer band.
    # Size the axis to what is drawn: the highest dot with --points, otherwise the highest
    # whisker (runs above the whisker are not drawn, so they must not stretch the axis).
    data_top = max(float(np.max(data[g])) if show_points else float(np.percentile(data[g], whis[1]))
                   for g in groups)
    if vms_value is not None:
        data_top = max(data_top, vms_value)
        for _c, _lo, hi in (vms_bands or []):
            data_top = max(data_top, hi)
    top = ymax if ymax is not None else data_top * 1.06
    ax.set_ylim(bottom=0, top=top)
    off_scale = vms_value is not None and vms_value >= top

    random_line = ax.axhline(1.0, color="#888", linestyle=":", linewidth=1.2, zorder=1,
                             label="random ordering (CAROB = 1)")
    median_handle, band_handles = None, []
    if vms_value is not None and vms_bands:
        # Nested bands, widest (palest) first; narrower bands are drawn darker.
        shown = [(c, lo, hi) for c, lo, hi in vms_bands if lo < top]
        alphas = np.linspace(0.13, 0.30, len(shown)) if len(shown) > 1 else [0.22]
        for (c, lo, hi), alpha in zip(shown, alphas):
            h = ax.axhspan(lo, hi, facecolor=VMS_COLOR, alpha=float(alpha), linewidth=0,
                           zorder=0.6, label=f"VMS central {c:g}% of runs ({lo:.2f}\u2013{hi:.2f})")
            band_handles.append(h)
        band_handles.reverse()          # legend: narrowest band first
    if vms_value is not None:
        runs = f", {vms_n_runs} runs" if vms_n_runs else ""
        vms_label = f"VMS median CAROB ({vms_value:.2f}{runs})"
        if off_scale:
            vms_label += " - off scale"
        median_handle = ax.axhline(vms_value, color=VMS_COLOR, linestyle="--", linewidth=1.5,
                                   alpha=0.9, zorder=2, label=vms_label)
        if off_scale:
            x_mid = (len(groups) + 1) / 2
            ax.plot([x_mid], [top], marker="^", color=VMS_COLOR, markersize=12,
                    clip_on=False, zorder=6, linestyle="none")
            ax.annotate(f"VMS median = {vms_value:.2f} (off scale)", xy=(x_mid, top),
                        xytext=(0, -12), textcoords="offset points", ha="center",
                        va="top", color=VMS_COLOR, fontsize=10, fontweight="bold")
    handles = ([median_handle] if median_handle is not None else []) + band_handles + [random_line]
    # Legend sits above the axes so it can never cover data or the VMS line.
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.01), ncol=2,
              fontsize=9, frameon=False)

    ax.set_xticks(pos)
    ax.set_xticklabels([f"{g}\n(n={len(data[g])} runs)" for g in groups], fontsize=10)
    ax.set_ylabel("CAROB: conjoint adjacency ratio over baseline\n(individual runs of one text)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    # The dot legend and the box definition hang BELOW the figure (savefig's tight
    # bounding box includes them), so they do not move the axes: the header legend and
    # plot area line up with violin_v2's chart when the two are placed side by side.
    y = 0.0
    if cats is not None and show_points and cat_colors:
        hs = [Line2D([0], [0], marker="o", ls="", color=c, label=str(k)) for k, c in cat_colors.items()]
        fig.legend(handles=hs, title=cat_title, loc="upper center", bbox_to_anchor=(0.5, 0.0),
                   ncol=len(hs), frameon=False, fontsize=9, title_fontsize=9)
        y = -0.085
    note = (f"box = {box[0]:g}th\u2013{box[1]:g}th percentile, whiskers = {whis[0]:g}th\u2013"
            f"{whis[1]:g}th, line = median"
            + ("; dots = individual runs" if show_points else ""))
    fig.text(0.5, y - 0.012, note, ha="center", va="top", fontsize=8.5, color="#555")
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    if png:
        fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", type=Path, nargs="?", default=None,
                    help="the single source .txt to run through sweep_v2 (text mode)")
    ap.add_argument("--results", type=Path, default=None,
                    help="existing sweep_v2 results CSV (results mode: no sweep is run)")
    ap.add_argument("--source-text", default=None,
                    help="which source_text to plot if the results file holds several")
    ap.add_argument("--vms-input", type=Path, default=None,
                    help="voynich_run_v2 results_full.csv (the VMS overlay)")
    ap.add_argument("--vms-bands", type=float, nargs="+", default=v.VMS_BANDS_DEFAULT,
                    metavar="PCT", help="central coverage (%%) of the nested VMS bands "
                    "(default 50 90, as in violin_v2)")
    ap.add_argument("--no-vms-bands", action="store_true", help="draw only the VMS median")
    ap.add_argument("--out-dir", type=Path, default=Path("boxplot_v2_out"))
    # filters / grid (forwarded to sweep_v2 in text mode; also applied as filters)
    ap.add_argument("--wpf", type=int, nargs="+", default=None,
                    help="words per folio. Text mode default: the VMS average (378) +/- 4, 8 = "
                         "370 374 378 382 386. Results mode: only these values "
                         "(text only; VMS is exempt)")
    ap.add_argument("--bpq", type=int, nargs="+", default=None,
                    help="quire constructions to run (text mode)")
    ap.add_argument("--metrics", nargs="+", choices=["hellinger", "js", "l1"], default=None,
                    help="only these distance metrics (text and VMS)")
    ap.add_argument("--best-cost-only", action="store_true",
                    help="only each cell's lowest-seriation-cost restart (text and VMS)")
    # text-mode sweep settings
    ap.add_argument("--vms-foliation", type=Path, default=VMS_FOLIATION_CSV,
                    help="VMS foliation CSV; its token count is the target window length")
    ap.add_argument("--length", type=int, default=None, help="window length in words (override)")
    ap.add_argument("--windows", type=int, default=N_WINDOWS_DEFAULT,
                    help="bootstrap windows if the text is longer than the window")
    ap.add_argument("--restarts", type=int, default=None,
                    help=f"SA restarts per metric (text mode default: {RESTARTS_DEFAULT})")
    ap.add_argument("--nmf-seeds", type=int, default=None)
    ap.add_argument("--sa-steps", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="recompute sweep results")
    # appearance
    ap.add_argument("--box", type=float, nargs=2, default=list(BOX_DEFAULT), metavar=("LO", "HI"),
                    help="box percentiles (default 25 75)")
    ap.add_argument("--whiskers", type=float, nargs=2, default=list(WHISKERS_DEFAULT),
                    metavar=("LO", "HI"), help="whisker percentiles (default 5 95)")
    ap.add_argument("--points", action="store_true",
                    help="also draw every individual run as a dot (off by default: the dots "
                         "are hard to read at small print sizes)")
    ap.add_argument("--no-points", action="store_true", help=argparse.SUPPRESS)   # old default
    ap.add_argument("--color-points-by", choices=["auto", "wpf", "metric", "group"], default="auto",
                    help="with --points: dot colour -- auto (words per folio if several, else "
                         "metric), wpf, metric, or the group colour")
    ap.add_argument("--ymax", type=float, default=None, help="y-axis upper limit (use the same on the violin chart)")
    ap.add_argument("--png", action="store_true", help="also write a PNG")
    ap.add_argument("--tail-report", action="store_true",
                    help="write tail_report.csv and print it: P(run >= VMS median) and the share "
                         "of runs inside the VMS bands, with cluster-aware 95%% intervals "
                         "(descriptive; needs --vms-input)")
    ap.add_argument("--tail-boot", type=int, default=4000, help="bootstrap replicates (default 4000)")
    ap.add_argument("--tail-seed", type=int, default=1, help="seed for the tail-report bootstrap")
    args = ap.parse_args()

    if (args.text is None) == (args.results is None):
        ap.error("give either the source text (text mode) or --results (results mode), not both/neither.")
    for lo, hi in (args.box, args.whiskers):
        if not 0 <= lo < hi <= 100:
            ap.error("--box / --whiskers need 0 <= LO < HI <= 100.")
    if not args.whiskers[0] <= args.box[0] or not args.box[1] <= args.whiskers[1]:
        ap.error("whiskers must be at least as wide as the box.")
    bad = [c for c in args.vms_bands if not 0 < c < 100]
    if bad:
        ap.error(f"--vms-bands values must be between 0 and 100 (got {bad}).")
    if args.tail_report and args.vms_input is None:
        ap.error("--tail-report compares the text with the VMS, so it needs --vms-input.")
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- obtain the results --------------------------------------------------
    plan = {}
    if args.text is not None:
        if not args.text.exists():
            ap.error(f"text not found: {args.text}")
        if not SWEEP_V2.exists():
            raise SystemExit(f"sweep_v2.py not found next to this script ({SWEEP_V2}).")
        import foliation as fol
        n_tok = len(fol.tokenize_clean(args.text.read_text(encoding="utf-8")))
        n_vms = vms_token_count(args.vms_foliation) if args.vms_foliation.exists() else None
        if args.length is not None:
            window = args.length
        elif n_vms is not None and n_tok >= n_vms:
            window = n_vms
        else:
            window = n_tok
        if window > n_tok:
            raise SystemExit(f"--length {window} exceeds the text's {n_tok} words.")
        print("=" * 70)
        print(f"Single-text box plot: {args.text.name}")
        print(f"  text length   : {n_tok} words")
        if n_vms is not None:
            print(f"  VMS length    : {n_vms} words")
        note = ""
        if window == n_tok and (n_vms is not None and n_tok != n_vms):
            note = (f"  (the whole text, as one window: {abs(n_vms - n_tok)} words, "
                    f"{abs(n_vms - n_tok) / n_vms:.1%}, {'fewer' if n_tok < n_vms else 'more'} than the VMS)")
        print(f"  window length : {window} words{note}")
        if args.wpf is None:
            if args.vms_foliation.exists():
                avg, n_t, n_f = vms_avg_words_per_folio(args.vms_foliation)
                centre = int(round(avg))
                wnote = (f"(VMS average {n_t:,} tokens / {n_f} folios = {avg:.2f} -> {centre}, "
                         f"{'/'.join(f'{o:+d}' for o in WPF_OFFSETS if o)})")
            else:
                centre = WPF_FALLBACK
                wnote = (f"(built-in VMS average {centre} {'/'.join(f'{o:+d}' for o in WPF_OFFSETS if o)}; "
                         f"the VMS CSV was not found)")
            args.wpf = [centre + o for o in WPF_OFFSETS]
        else:
            wnote = "(from --wpf)"
        print(f"  words/folio   : {', '.join(map(str, args.wpf))}  {wnote}")
        if args.restarts is None:
            args.restarts = RESTARTS_DEFAULT
            rnote = "(default; sweep_v2's own is 8)"
        else:
            rnote = "(from --restarts)"
        print(f"  SA restarts   : {args.restarts} per metric per folio size  {rnote}")
        print(f"  runs per box  : {len(args.wpf)} size(s) x {len(args.metrics) if args.metrics else 3} "
              f"metric(s) x {args.restarts} restart(s) = "
              f"{len(args.wpf) * (len(args.metrics) if args.metrics else 3) * args.restarts}"
              f" x {1 if window == n_tok else args.windows} window(s)")
        print("=" * 70, flush=True)
        results_path = run_sweep(args.text, out_dir / "sweep", window, args.windows, args)
        source = args.text.stem
        plan = dict(mode="text", text=str(args.text), n_words=n_tok, window_len=window)
    else:
        results_path, source = args.results, args.source_text
        plan = dict(mode="results", results=str(args.results))

    df = v.read_results(results_path)
    names = sorted(df["source_text"].unique())
    if source is None:
        if len(names) > 1:
            raise SystemExit(f"{results_path} holds {len(names)} texts ({', '.join(names[:8])}"
                             f"{', ...' if len(names) > 8 else ''}); choose one with --source-text.")
        source = names[0]
    if source not in names:
        raise SystemExit(f"source_text {source!r} not found in {results_path}; available: {names[:10]}")
    df = df[df["source_text"] == source].copy()
    n_rows_text = len(df)
    df = v.apply_filters(df, args.wpf, args.metrics, args.best_cost_only)
    if df.empty:
        raise SystemExit("No rows left after filtering.")

    df["bpq_label"] = df["bifolios_per_quire"].map(v.bpq_label)
    bpqs = sorted(df["bifolios_per_quire"].unique())
    groups = [v.bpq_label(n) for n in bpqs]
    data = {g: df.loc[df["bpq_label"] == g, v.VALUE_COL].to_numpy(dtype=float) for g in groups}

    # ---- dot colouring ---------------------------------------------------------
    cats, cat_colors, cat_title = None, None, None
    mode = args.color_points_by
    if mode == "auto":
        n_wpf = df["words_per_folio"].nunique() if "words_per_folio" in df.columns else 0
        mode = "wpf" if n_wpf > 1 else "metric"
    if mode == "wpf" and "words_per_folio" in df.columns \
            and df["words_per_folio"].notna().any():
        vals = sorted(df["words_per_folio"].dropna().unique())
        cat_colors = {f"{int(x)}": _WPF_COLORS[i % len(_WPF_COLORS)] for i, x in enumerate(vals)}
        cats = {g: df.loc[df["bpq_label"] == g, "words_per_folio"].map(lambda x: f"{int(x)}").to_numpy()
                for g in groups}
        cat_title = "dots: words per folio"
    elif mode == "metric" and "metric" in df.columns:
        vals = sorted(df["metric"].unique())
        cat_colors = {m: _METRIC_COLORS.get(m, "#555") for m in vals}
        cats = {g: df.loc[df["bpq_label"] == g, "metric"].to_numpy() for g in groups}
        cat_title = "dots: distance metric"
    if cats is not None and len(cat_colors) == 1:        # a single category adds nothing
        cats, cat_colors, cat_title = None, None, None

    # ---- VMS (same handling as violin_v2) --------------------------------------
    vms_value, vms_bands, vms_n, vms_df = None, None, None, None
    if args.vms_input is not None:
        vms = v.read_results(args.vms_input)
        vms = v.apply_filters(vms, args.wpf, args.metrics, args.best_cost_only,
                              exempt_missing_wpf=True)
        if vms.empty:
            print("  WARNING: all VMS rows were removed by the filters; drawing without the VMS.")
        else:
            vals = vms[v.VALUE_COL].to_numpy(dtype=float)
            vms_df = vms
            vms_value, vms_n = float(np.median(vals)), len(vals)
            stat_rows = [("n_runs", float(len(vals))), ("min", float(vals.min())),
                         ("median", vms_value), ("max", float(vals.max()))]
            if "metric" in vms.columns:
                stat_rows += [(f"median[metric={m}]", float(g[v.VALUE_COL].median()))
                              for m, g in vms.groupby("metric")]
            if not args.no_vms_bands:
                if len(vals) >= v.MIN_RUNS_FOR_BANDS:
                    vms_bands = v.vms_band_table(vals, args.vms_bands)
                    for c, lo, hi in vms_bands:
                        stat_rows += [(f"central_{c:g}pct_lo", lo), (f"central_{c:g}pct_hi", hi)]
                else:
                    print(f"  NOTE: only {len(vals)} VMS run(s) (< {v.MIN_RUNS_FOR_BANDS}); "
                          f"percentile bands not drawn.")
            pd.DataFrame(stat_rows, columns=["statistic", "value"]).to_csv(
                out_dir / "vms_percentiles.csv", index=False)

    # ---- summaries ----------------------------------------------------------------
    box, whis = tuple(args.box), tuple(args.whiskers)
    tab = percentile_table(data, groups, box, whis)
    tab.to_csv(out_dir / "boxplot_percentiles.csv", index=False)
    breakdown_table(df).to_csv(out_dir / "boxplot_breakdown.csv", index=False)

    print(f"\nText: {source}   ({len(df)} of {n_rows_text} runs after filters)")
    with pd.option_context("display.float_format", lambda x: f"{x:.2f}", "display.width", 160):
        print(tab.to_string(index=False))
    if vms_value is not None:
        print(f"\nVMS median CAROB {vms_value:.2f} over {vms_n} runs"
              + ("" if not vms_bands else "   (" + "   ".join(
                  f"central {c:g}%: {lo:.2f}-{hi:.2f}" for c, lo, hi in vms_bands) + ")"))
        if args.ymax is not None and vms_value >= args.ymax:
            print(f"  WARNING: VMS median ({vms_value:.2f}) is at/above --ymax ({args.ymax:g}); "
                  f"it is marked at the top edge as off scale.")

    if args.tail_report:
        if vms_df is None:
            print("  WARNING: --tail-report skipped: no VMS rows left after the filters.")
        else:
            cov = list(args.vms_bands) if not args.no_vms_bands else list(v.VMS_BANDS_DEFAULT)
            tail = tail_report(df, groups, vms_df, vms_value, cov, args.tail_boot, args.tail_seed)
            tail.round(5).to_csv(out_dir / "tail_report.csv", index=False)
            print_tail_report(tail, cov)
            print(f"\nTail report -> {out_dir / 'tail_report.csv'}")

    svg = out_dir / "boxplot_single_text.svg"
    make_boxplot(data, cats, cat_colors, cat_title, groups, vms_value, svg, box=box, whis=whis,
                 ymax=args.ymax, vms_n_runs=vms_n, vms_bands=vms_bands,
                 show_points=args.points and not args.no_points, png=args.png)
    print(f"\nChart -> {svg}")

    (out_dir / "run_settings.json").write_text(json.dumps(dict(
        source_text=source, **plan, vms_input=str(args.vms_input) if args.vms_input else None,
        filters=dict(wpf=args.wpf, metrics=args.metrics, best_cost_only=args.best_cost_only),
        box=list(box), whiskers=list(whis), vms_bands=None if args.no_vms_bands else args.vms_bands,
        runs_plotted=int(len(df)), groups=groups,
        tail_report=dict(enabled=bool(args.tail_report), boot=args.tail_boot, seed=args.tail_seed), created=time.strftime("%Y-%m-%d %H:%M:%S"),
        versions=dict(pandas=pd.__version__, numpy=np.__version__,
                      matplotlib=matplotlib.__version__)), indent=2))
    print("Done.")


if __name__ == "__main__":
    main()
