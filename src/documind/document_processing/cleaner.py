"""Unicode normalisation and whitespace cleanup for extracted text."""

from __future__ import annotations

import re
import unicodedata

_LIGATURES = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
}
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f​﻿]")
_SPACES = re.compile(r"[ \t  -   　]+")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(?=[a-zà-ỹ])")
_MANY_NEWLINES = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """Return NFC-normalised text with control chars removed and whitespace tidied.

    NFC (not NFKC) is used so Vietnamese diacritics are composed consistently
    without rewriting compatibility characters in the quoted evidence.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    for lig, repl in _LIGATURES.items():
        text = text.replace(lig, repl)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("­", "")
    text = _CONTROL_CHARS.sub("", text)
    text = _SPACES.sub(" ", text)
    text = _HYPHEN_BREAK.sub(r"\1", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _MANY_NEWLINES.sub("\n\n", text)
    return text.strip()


def clean_paragraph(text: str) -> str:
    """Normalise and join hard-wrapped lines of a single paragraph into one line."""
    text = normalize_text(text)
    return " ".join(part for part in text.split("\n") if part).strip()
