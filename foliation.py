"""
foliation.py
============
Library module for synthetic manuscript foliation.

Two entry points:
  - generate_foliation(text_path, words_per_folio, bifolios_per_quire)
      Tokenizes a plain-text file and assigns folio/bifolio/quire structure.
  - load_predetermined_foliation(csv_path)
      Loads an existing foliation CSV (e.g. for the Voynich Manuscript)
      and returns the same schema as generate_foliation.

Both return a DataFrame with columns:
    doc_id, token, bifolio_id, quire, folio_in_quire, side, bifolio_global

'bifolio_global' is a unique integer identifier for each bifolio across all
quires, used by the sweep to evaluate adjacency in the recovered ordering.

Also exports:
  - random_baseline(foliation_df)
      Analytical expected bifolio-adjacency rate under a random ordering.
"""

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------

def strip_diacritics(s: str) -> str:
    """NFD-normalize and remove combining diacritical marks."""
    normalized = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def tokenize_clean(text: str) -> list[str]:
    """Lowercase, strip diacritics, return alphabetic tokens only."""
    text = strip_diacritics(text)
    text = text.lower()
    return re.findall(r"[a-z]+", text)


# ---------------------------------------------------------------------------
# Structural assignment helpers
# ---------------------------------------------------------------------------

def _assign_structure(word_index: int,
                      words_per_folio: int,
                      bifolios_per_quire: int) -> tuple[int, int, int, int]:
    """
    Given a 1-based word index, return (folio, bifolio_in_quire, quire, folio_in_quire).

    Folio numbering is global (1-based, sequential).
    Within each quire of F = 2 * bifolios_per_quire folios, bifolio assignment
    reflects physical nesting: folios 1 and F share a sheet, 2 and F-1 share a
    sheet, and so on.
    """
    folio = (word_index - 1) // words_per_folio + 1
    folios_per_quire = 2 * bifolios_per_quire
    quire = (folio - 1) // folios_per_quire + 1
    folio_in_quire = (folio - 1) % folios_per_quire + 1
    bifolio_in_quire = min(folio_in_quire,
                           folios_per_quire - folio_in_quire + 1)
    return folio, bifolio_in_quire, quire, folio_in_quire


def _recto_verso(word_index: int, words_per_folio: int) -> str:
    """
    Label a word as 'recto' or 'verso' within its folio.
    First floor(words_per_folio / 2) words are recto; remainder are verso.
    """
    pos_in_folio = (word_index - 1) % words_per_folio + 1
    half = words_per_folio // 2
    return "recto" if pos_in_folio <= half else "verso"


# ---------------------------------------------------------------------------
# Primary entry point: generative foliation
# ---------------------------------------------------------------------------

def generate_foliation(text_path: Path | str,
                       words_per_folio: int,
                       bifolios_per_quire: int) -> pd.DataFrame:
    """
    Read a plain-text file, tokenize it, and assign manuscript structure.

    Parameters
    ----------
    text_path : path-like
        UTF-8 plain-text input file.
    words_per_folio : int
        Number of tokens assigned to each folio.
    bifolios_per_quire : int
        Number of bifolios nested within each quire.

    Returns
    -------
    DataFrame with columns:
        doc_id          int   global folio number (1-based)
        token           str   cleaned word
        bifolio_id      int   bifolio within quire (1 .. bifolios_per_quire)
        quire           int   quire number (1-based)
        folio_in_quire  int   folio position within quire (1-based)
        side            str   'recto' or 'verso'
        bifolio_global  int   unique bifolio id across all quires

    Raises
    ------
    ValueError if the file contains no usable tokens.
    RuntimeWarning if the last folio is shorter than words_per_folio.
    """
    text_path = Path(text_path)
    text = text_path.read_text(encoding="utf-8")
    words = tokenize_clean(text)

    if not words:
        raise ValueError(f"No valid tokens found in {text_path}")

    n_words = len(words)
    n_full_folios = n_words // words_per_folio
    remainder = n_words % words_per_folio
    if remainder > 0:
        import warnings
        warnings.warn(
            f"Last folio has only {remainder} tokens (expected {words_per_folio}). "
            "It will be included but its topic distribution may be noisier.",
            RuntimeWarning,
            stacklevel=2,
        )

    rows = []
    for i, w in enumerate(words, start=1):
        folio, bifolio_id, quire, folio_in_quire = _assign_structure(
            i, words_per_folio, bifolios_per_quire
        )
        side = _recto_verso(i, words_per_folio)
        # Unique bifolio across all quires
        bifolio_global = (quire - 1) * bifolios_per_quire + bifolio_id
        rows.append({
            "doc_id":         folio,
            "token":          w,
            "bifolio_id":     bifolio_id,
            "quire":          quire,
            "folio_in_quire": folio_in_quire,
            "side":           side,
            "bifolio_global": bifolio_global,
        })

    df = pd.DataFrame(rows)

    # Attach a synthetic mg_section (quire number as string) so the downstream
    # pipeline's section-aware code has something to work with if needed.
    df["mg_section"] = df["quire"].astype(str)

    return df


# ---------------------------------------------------------------------------
# Secondary entry point: predetermined foliation (Voynich / custom CSV)
# ---------------------------------------------------------------------------

_REQUIRED_PREDETERMINED = {"doc_id", "token", "bifolio_global"}

def load_predetermined_foliation(csv_path: Path | str) -> pd.DataFrame:
    """
    Load an existing foliation CSV and return it in the standard schema.

    The CSV must contain at minimum: doc_id, token, bifolio_global.
    Optional columns (bifolio_id, quire, folio_in_quire, side, mg_section)
    are passed through if present; missing optional columns are filled with NA.

    Parameters
    ----------
    csv_path : path-like
        Path to the predetermined foliation CSV.

    Returns
    -------
    DataFrame conforming to the same schema as generate_foliation.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    missing = _REQUIRED_PREDETERMINED - set(df.columns)
    if missing:
        raise ValueError(
            f"Predetermined foliation CSV is missing required columns: {missing}"
        )

    optional = ["bifolio_id", "quire", "folio_in_quire", "side", "mg_section"]
    for col in optional:
        if col not in df.columns:
            df[col] = pd.NA

    # Ensure doc_id and bifolio_global are integers
    df["doc_id"] = df["doc_id"].astype(int)
    df["bifolio_global"] = df["bifolio_global"].astype(int)

    return df[["doc_id", "token", "bifolio_id", "quire",
               "folio_in_quire", "side", "bifolio_global", "mg_section"]]


# ---------------------------------------------------------------------------
# Baseline: analytical random adjacency rate
# ---------------------------------------------------------------------------

def random_baseline(foliation_df: pd.DataFrame) -> float:
    """
    Expected fraction of adjacent pairs in a RANDOM ordering that share a
    bifolio, computed analytically.

    For a sequence of n folios drawn uniformly at random without replacement,
    the expected number of adjacent bifolio-mate pairs equals:

        sum over each bifolio b: C(size_b, 2) * 2 / (n - 1)

    where C(size_b, 2) is the number of pairs within bifolio b, and the
    factor 2/(n-1) is the probability that any given pair is adjacent in a
    random permutation of n items (both orderings count).

    Dividing by (n - 1) total adjacent pairs gives the expected rate.

    Parameters
    ----------
    foliation_df : DataFrame
        Output of generate_foliation or load_predetermined_foliation.
        Must contain doc_id and bifolio_global columns (one row per token;
        we first reduce to one row per folio).

    Returns
    -------
    float in [0, 1]
    """
    # One row per folio
    folio_df = (foliation_df[["doc_id", "bifolio_global"]]
                .drop_duplicates(subset="doc_id"))

    n = folio_df["doc_id"].nunique()
    if n < 2:
        return 0.0

    # Number of folios per bifolio
    sizes = folio_df.groupby("bifolio_global")["doc_id"].count()

    # Expected adjacent bifolio-mate pairs
    expected_adjacent = sum(
        s * (s - 1) / 2 * 2 / (n - 1)
        for s in sizes
    )

    return float(expected_adjacent / (n - 1))


# ---------------------------------------------------------------------------
# Convenience: summarise a foliation in human-readable form
# ---------------------------------------------------------------------------

def foliation_summary(foliation_df: pd.DataFrame) -> dict:
    """Return a dict of key counts for logging/debugging."""
    folio_df = foliation_df[["doc_id", "bifolio_global", "quire"]].drop_duplicates("doc_id")
    return {
        "n_tokens":    len(foliation_df),
        "n_folios":    folio_df["doc_id"].nunique(),
        "n_bifolios":  folio_df["bifolio_global"].nunique(),
        "n_quires":    folio_df["quire"].nunique() if "quire" in folio_df.columns else "N/A",
    }
