"""
perseus_orchestrator.py
========================
Scales up the synthetic-collation baseline (the 8 hand-picked texts in the
original study) using Perseus Digital Library's "canonical" TEI-XML text
repositories, then hands the result to sweep.py / foliation.py unchanged.

WHAT THIS DOES
--------------
1. Downloads each target Perseus repo as a tarball from codeload.github.com
   (fast, single request, no git history, no GitHub API rate limits) and
   caches it locally.
2. Walks every XML file, keeping only ORIGINAL-LANGUAGE editions (not English
   translations) by filename convention, e.g.
       phi0959.phi010.perseus-lat2.xml   -> keep  (Latin original)
       phi0959.phi010.perseus-eng2.xml   -> skip  (English translation)
       .../__cts__.xml                   -> skip  (metadata, not text)
3. Parses each candidate with lxml (namespace-agnostic, since Perseus mixes
   TEI P4 and P5 files), strips apparatus criticus variant readings, notes,
   and bibliographic refs, and extracts running text from <body>.
4. Tokenizes with the EXACT SAME function foliation.py uses
   (fol.tokenize_clean: NFD-strip diacritics, lowercase, alphabetic-only)
   so the word count used for filtering matches what the sweep will actually
   see, and filters out anything under --min-words.
5. Deduplicates by (author, title), keeping the longest edition when a work
   exists in more than one edition.
6. Writes each surviving text as a plain-text file into the same TEXTS_DIR
   sweep.py already reads from (default "texts/"), alongside a manifest CSV
   of every file considered and why it was kept or dropped.

USAGE
-----
    # Maximal: pull everything Perseus has, no per-language cap
    python perseus_orchestrator.py

    # Practical subset for a first test run
    python perseus_orchestrator.py --max-per-repo 10 --min-words 4000

    # Only Latin + Greek, keep existing texts/ dir contents
    python perseus_orchestrator.py --repos canonical-latinLit canonical-greekLit

After this script finishes, run sweep.py exactly as before -- it just sees a
bigger TEXTS_DIR. See the printed "compute cost" warning at the end for
why you probably want --max-per-repo on a first pass.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import tarfile
import unicodedata
import urllib.request
from pathlib import Path

from lxml import etree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import foliation as fol  # reuse the EXACT tokenizer sweep.py's pipeline uses

# =============================================================================
# CONFIGURATION
# =============================================================================

# repo_name -> (original-language filename infixes, human label, script)
# Perseus encodes language in the filename right after "perseus-", e.g.
# "...perseus-lat1.xml". English translations use "-eng*" regardless of repo,
# so excluding "-eng" and keeping only these infixes selects original-language
# editions.
#
# IMPORTANT — tokenizer compatibility:
# foliation.py's tokenize_clean() does NFD-normalize + lowercase + keep only
# [a-z]. NFD only decomposes PRECOMPOSED LATIN letters (e.g. e-acute -> e +
# combining accent); it does NOT decompose non-Latin scripts, and it does NOT
# decompose Latin-adjacent letters that aren't precomposed accented Latin
# letters (thorn þ, eth ð, ash æ). Concretely:
#   - Greek script            -> every token is dropped (0 usable words)
#   - Old Norse / Old English -> words containing þ/ð/æ are SHATTERED into
#                                 fragments ("þræll" -> "r", "ll"), which
#                                 silently corrupts the topic-modeling input
#                                 without erroring or looking obviously wrong
# "script_safe" below marks which repos are safe to run through the existing
# tokenizer UNCHANGED. Unsafe repos are excluded by default; see
# --include-unsafe-scripts to opt in (loudly warned).
REPOS: dict[str, dict] = {
    "canonical-latinLit": {"lang_infixes": ("lat",), "label": "Latin",       "script_safe": True},
    "canonical-itaLit":   {"lang_infixes": ("ita",), "label": "Italian",     "script_safe": True},
    "canonical-greekLit": {"lang_infixes": ("grc",), "label": "Greek",       "script_safe": False},
    "canonical-angLit":   {"lang_infixes": ("ang",), "label": "Old English", "script_safe": False},
    "canonical-norseLit": {"lang_infixes": ("non",), "label": "Old Norse",   "script_safe": False},
    "canonical-farsiLit": {"lang_infixes": ("far", "per"), "label": "Persian", "script_safe": False},
}

SCRIPT_SAFE_REPOS = [r for r, c in REPOS.items() if c["script_safe"]]

GITHUB_OWNER = "PerseusDL"
CODELOAD_URL = "https://codeload.github.com/{owner}/{repo}/tar.gz/refs/heads/master"

DEFAULT_CACHE_DIR = Path("perseus_cache")
DEFAULT_TEXTS_DIR = Path("texts")          # matches sweep.py's TEXTS_DIR
DEFAULT_MIN_WORDS = 4000                   # floor for inclusion (post-clean)

# TEI elements whose text should NOT be included in the running text.
# rdg = apparatus criticus variant readings (we keep the main <lem> reading
# implicitly by just not descending into <rdg>); note/bibl/ref are editorial
# apparatus, not the text itself.
_STRIP_TAGS = {"note", "bibl", "ref", "rdg", "app"}
# For <app>, if a <lem> child exists we still want ITS text -- so we strip
# rdg/note/bibl individually rather than the whole <app>, except when no lem
# is present (handled in code below).


# =============================================================================
# STEP 1 — DOWNLOAD + CACHE
# =============================================================================

def download_and_extract(repo: str, cache_dir: Path) -> Path:
    """Download a Perseus repo tarball (if not already cached) and extract it."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    tarball = cache_dir / f"{repo}.tar.gz"
    extracted_root = cache_dir / f"{repo}-master"

    if extracted_root.exists():
        print(f"  [{repo}] already extracted, skipping download")
        return extracted_root

    if not tarball.exists():
        url = CODELOAD_URL.format(owner=GITHUB_OWNER, repo=repo)
        print(f"  [{repo}] downloading {url} ...", end=" ", flush=True)
        try:
            urllib.request.urlretrieve(url, tarball)
        except Exception as e:
            print(f"FAILED ({e})")
            return extracted_root  # may not exist; caller checks
        print(f"done ({tarball.stat().st_size / 1e6:.1f} MB)")

    print(f"  [{repo}] extracting ...", end=" ", flush=True)
    with tarfile.open(tarball) as tf:
        tf.extractall(cache_dir, filter="data")
    print("done")
    return extracted_root


# =============================================================================
# STEP 2 — DISCOVER ORIGINAL-LANGUAGE FILES
# =============================================================================

_FNAME_RE = re.compile(r"\.perseus-([a-z]+)\d*\.xml$", re.IGNORECASE)


def discover_candidate_files(repo_root: Path, lang_infixes: tuple[str, ...]) -> list[Path]:
    """Find XML files whose filename infix matches an original-language code."""
    if not repo_root.exists():
        return []
    out = []
    for xml_path in repo_root.rglob("*.xml"):
        if xml_path.name == "__cts__.xml":
            continue
        m = _FNAME_RE.search(xml_path.name)
        if not m:
            continue
        infix = m.group(1).lower()
        if infix in lang_infixes:
            out.append(xml_path)
    return out


# =============================================================================
# STEP 3 — PARSE TEI -> PLAIN TEXT
# =============================================================================

def _local(tag) -> str:
    """Strip XML namespace from an lxml tag name."""
    return etree.QName(tag).localname if isinstance(tag, str) else tag


def extract_plaintext(xml_path: Path) -> tuple[str | None, str | None, str]:
    """
    Parse a TEI file and return (author, title, running_text).
    Namespace-agnostic: Perseus mixes un-namespaced TEI P4 and namespaced P5.
    Returns ("", "", "") on parse failure.
    """
    try:
        parser = etree.XMLParser(recover=True, huge_tree=True)
        tree = etree.parse(str(xml_path), parser)
    except Exception:
        return None, None, ""

    root = tree.getroot()

    def find_one(tag_name: str):
        results = root.xpath(f".//*[local-name()='{tag_name}']")
        return results[0] if results else None

    title_el = find_one("title")
    author_el = find_one("author")
    title = (title_el.text or "").strip() if title_el is not None else ""
    author = (author_el.text or "").strip() if author_el is not None else ""

    body = find_one("body")
    target = body if body is not None else find_one("text")
    if target is None:
        return author, title, ""

    # Remove apparatus/editorial noise before extracting text, working on a
    # deep copy so we don't mutate anything unexpectedly.
    target = etree.fromstring(etree.tostring(target))
    for tag in ("note", "bibl", "ref", "rdg"):
        for el in target.xpath(f".//*[local-name()='{tag}']"):
            el.getparent().remove(el)

    text = "".join(target.itertext())
    text = re.sub(r"\s+", " ", text).strip()
    return author, title, text


# =============================================================================
# STEP 4 — FILTER, DEDUPE, WRITE
# =============================================================================

def slugify(*parts: str, fallback: str) -> str:
    raw = "_".join(p for p in parts if p) or fallback
    raw = unicodedata.normalize("NFD", raw)
    raw = "".join(c for c in raw if unicodedata.category(c) != "Mn")
    raw = re.sub(r"[^a-zA-Z0-9]+", "_", raw).strip("_").lower()
    return raw[:80] or fallback


def process_repo(repo: str, cfg: dict, cache_dir: Path,
                  min_words: int, max_texts: int | None,
                  manifest_rows: list[dict]) -> dict[str, dict]:
    """
    Returns dict: dedupe_key -> {author, title, text, word_count, source_path}
    for texts from this repo that pass the length filter (pre-dedup/cap).
    """
    label = cfg["label"]
    repo_root = download_and_extract(repo, cache_dir)
    candidates = discover_candidate_files(repo_root, cfg["lang_infixes"])
    print(f"  [{repo}] {len(candidates)} original-language file(s) found")

    kept: dict[str, dict] = {}
    for xml_path in candidates:
        author, title, raw_text = extract_plaintext(xml_path)
        if not raw_text:
            manifest_rows.append(dict(
                repo=repo, path=str(xml_path), language=label,
                author=author, title=title, word_count=0,
                included=False, reason="parse_failed_or_empty",
            ))
            continue

        words = fol.tokenize_clean(raw_text)
        wc = len(words)

        # Diagnostic: what fraction of characters in the raw text are outside
        # the Latin-alphabet range the tokenizer can handle? A high fraction
        # combined with a "reasonable" wc is the signature of silent word
        # fragmentation (thorn/eth/ash), not clean tokenization.
        latin_chars = len(re.findall(r"[A-Za-z]", raw_text))
        total_letters = len(re.findall(r"\w", raw_text, re.UNICODE))
        non_latin_frac = 1 - (latin_chars / total_letters) if total_letters else 0.0

        if wc < min_words:
            manifest_rows.append(dict(
                repo=repo, path=str(xml_path), language=label,
                author=author, title=title, word_count=wc,
                included=False, reason=f"below_min_words({min_words})",
            ))
            continue

        key = slugify(label, author, title, fallback=xml_path.stem)
        entry = dict(author=author, title=title, text=raw_text,
                     word_count=wc, source_path=str(xml_path), label=label,
                     non_latin_frac=non_latin_frac)

        if key not in kept or wc > kept[key]["word_count"]:
            if key in kept:
                manifest_rows.append(dict(
                    repo=repo, path=kept[key]["source_path"], language=label,
                    author=kept[key]["author"], title=kept[key]["title"],
                    word_count=kept[key]["word_count"], included=False,
                    reason="superseded_by_longer_edition",
                ))
            kept[key] = entry
        else:
            manifest_rows.append(dict(
                repo=repo, path=str(xml_path), language=label,
                author=author, title=title, word_count=wc,
                included=False, reason="duplicate_shorter_edition",
            ))

    if max_texts is not None and len(kept) > max_texts:
        # Keep the longest `max_texts` texts (bias toward more topical signal
        # per document rather than an arbitrary cut).
        ranked = sorted(kept.items(), key=lambda kv: -kv[1]["word_count"])
        keep_keys = {k for k, _ in ranked[:max_texts]}
        for k, v in kept.items():
            if k not in keep_keys:
                manifest_rows.append(dict(
                    repo=repo, path=v["source_path"], language=label,
                    author=v["author"], title=v["title"],
                    word_count=v["word_count"], included=False,
                    reason=f"exceeds_max_per_repo({max_texts})",
                ))
        kept = {k: v for k, v in kept.items() if k in keep_keys}

    for k, v in kept.items():
        manifest_rows.append(dict(
            repo=repo, path=v["source_path"], language=label,
            author=v["author"], title=v["title"],
            word_count=v["word_count"], included=True, reason="kept",
        ))

    print(f"  [{repo}] {len(kept)} text(s) kept "
          f"(>= {min_words} words, deduped{f', capped at {max_texts}' if max_texts else ''})")

    if kept and not cfg["script_safe"]:
        avg_frac = sum(v["non_latin_frac"] for v in kept.values()) / len(kept)
        print(f"  [{repo}] !! TOKENIZER WARNING: this repo's script is not "
              f"reliably handled by foliation.py's tokenizer.")
        print(f"  [{repo}]    avg. {avg_frac:.1%} of letters in kept texts are "
              f"outside plain a-z after NFD-normalization.")
        print(f"  [{repo}]    These texts will still be WRITTEN if you passed "
              f"--include-unsafe-scripts, but tokenize_clean() will silently")
        print(f"  [{repo}]    drop (Greek) or fragment (thorn/eth/ash in Old "
              f"Norse/English) words -- inspect before trusting results.")

    return kept


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repos", nargs="+", default=SCRIPT_SAFE_REPOS,
                         choices=list(REPOS.keys()),
                         help=f"Which Perseus repos to pull from "
                              f"(default: script-safe only -- {SCRIPT_SAFE_REPOS})")
    parser.add_argument("--include-unsafe-scripts", action="store_true",
                         help="Allow non-Latin-script repos (Greek, Old Norse, Old "
                              "English, Persian) even though foliation.py's tokenizer "
                              "was not built for them (see module docstring). Off by "
                              "default because it silently corrupts/drops tokens.")
    parser.add_argument("--min-words", type=int, default=DEFAULT_MIN_WORDS,
                         help=f"Minimum cleaned word count to keep a text (default {DEFAULT_MIN_WORDS})")
    parser.add_argument("--max-per-repo", type=int, default=None,
                         help="Cap the number of texts kept per repo (default: no cap -- "
                              "pull everything that passes the length filter)")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
                         help="Where to cache downloaded/extracted repos")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_TEXTS_DIR,
                         help="Output directory for plain-text files (matches sweep.py's TEXTS_DIR)")
    parser.add_argument("--dry-run", action="store_true",
                         help="Report counts only; don't write any text files")
    args = parser.parse_args()

    unsafe_selected = [r for r in args.repos if not REPOS[r]["script_safe"]]
    if unsafe_selected and not args.include_unsafe_scripts:
        print("The following selected repos use scripts foliation.py's tokenizer "
              "cannot handle correctly:")
        for r in unsafe_selected:
            print(f"  - {r} ({REPOS[r]['label']})")
        print("Re-run with --include-unsafe-scripts to include them anyway "
              "(not recommended without fixing tokenize_clean first), or drop "
              "them from --repos.")
        return

    print("=" * 70)
    print("Perseus corpus orchestrator")
    print(f"  Repos       : {', '.join(args.repos)}")
    print(f"  Min words   : {args.min_words}")
    print(f"  Max/repo    : {args.max_per_repo or 'unlimited'}")
    print(f"  Cache dir   : {args.cache_dir}")
    print(f"  Out dir     : {args.out_dir}")
    print("=" * 70)

    manifest_rows: list[dict] = []
    all_kept: dict[str, dict] = {}

    for repo in args.repos:
        cfg = REPOS[repo]
        print(f"\n{repo} ({cfg['label']})")
        kept = process_repo(repo, cfg, args.cache_dir, args.min_words,
                             args.max_per_repo, manifest_rows)
        for k, v in kept.items():
            # Global de-dup across repos too (unlikely but cheap to guard)
            all_kept.setdefault(k, v)

    # ---- Write manifest ----
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir.parent / "perseus_manifest.csv" if args.out_dir.parent != Path(".") \
        else Path("perseus_manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "repo", "path", "language", "author", "title",
            "word_count", "included", "reason",
        ])
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"\nManifest -> {manifest_path}  ({len(manifest_rows)} files considered)")

    # ---- Per-language summary ----
    by_lang: dict[str, int] = {}
    for v in all_kept.values():
        by_lang[v["label"]] = by_lang.get(v["label"], 0) + 1
    print("\nTexts kept by language:")
    for lang, n in sorted(by_lang.items(), key=lambda x: -x[1]):
        print(f"  {lang:15s} {n}")
    print(f"  {'TOTAL':15s} {len(all_kept)}")

    # ---- Write plain-text files ----
    if args.dry_run:
        print("\n--dry-run set: no text files written.")
    else:
        for key, v in all_kept.items():
            out_path = args.out_dir / f"{key}.txt"
            out_path.write_text(v["text"], encoding="utf-8")
        print(f"\nWrote {len(all_kept)} text file(s) -> {args.out_dir}/")

    # ---- Compute-cost warning ----
    n_texts = len(all_kept)
    combos_per_text = 3 * 5 * 3   # wpf * bpq * metric, per sweep.py's default grid
    total_combos = n_texts * combos_per_text
    total_runs = total_combos * 8  # restarts
    print("\n" + "=" * 70)
    print("COMPUTE COST ESTIMATE for sweep.py's default grid")
    print(f"  {n_texts} texts x {combos_per_text} (wpf x bpq x metric) combos "
          f"= {total_combos} combos")
    print(f"  x 8 restarts = {total_runs} total SA seriations")
    print(f"  Each combo also fits a 55-model NMF ensemble from scratch.")
    if n_texts > 20:
        print(f"  This is a LOT more than the original study's 8 texts "
              f"({n_texts / 8:.0f}x). Consider --max-per-repo for a first pass,")
        print(f"  or reduce sweep.py's WORDS_PER_FOLIO_VALUES / N_RESTARTS "
              f"before running the full grid.")
    print("=" * 70)


if __name__ == "__main__":
    main()
