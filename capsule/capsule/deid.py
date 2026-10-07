"""The deliberately small set of source metadata allowed in a capsule."""

from __future__ import annotations

import re


_SAFE_DESCRIPTIONS = {
    "CT HEAD": "CT head", "HEAD CT": "CT head", "MR BRAIN": "MR brain",
    "MRI BRAIN": "MR brain", "MR T1": "MR T1", "MR T2": "MR T2",
    "MR FLAIR": "MR FLAIR", "MR DWI": "MR DWI", "MR ADC": "MR ADC",
    "SYNTHETIC HEAD CT": "Synthetic head CT",
    "SYNTHETIC HEAD MR T1": "Synthetic head MR T1",
}
_SAFE_SEQUENCES = {"SE", "IR", "GR", "EP", "RM", "T1_SE", "T2_SE", "FLAIR", "DWI", "ADC"}


def safe_description(value: object) -> str:
    """Discard arbitrary scanner free text, which often contains identifiers."""
    text = str(value or "").strip().upper()
    return _SAFE_DESCRIPTIONS.get(text, "(omitted)")


def safe_sequence(value: object) -> str:
    text = str(value or "").strip().upper()
    return text if text in _SAFE_SEQUENCES else "(omitted)"


def study_year(value: object) -> int | None:
    text = str(value or "")
    if re.fullmatch(r"\d{8}", text):
        year = int(text[:4])
        if 1900 <= year <= 2100:
            return year
    return None


_WEIGHTINGS = ("FLAIR", "T1", "T2", "PD", "DWI", "ADC", "SWI")


def mr_weighting(value: object) -> str | None:
    """Return one fixed weighting token found in free text, never the text itself."""
    tokens = set(re.split(r"[^A-Z0-9]+", str(value or "").upper()))
    return next((weighting for weighting in _WEIGHTINGS if weighting in tokens), None)
