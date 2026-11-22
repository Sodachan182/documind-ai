"""Application-controlled citation IDs and validation of model output.

The model only ever sees IDs ``S1..Sn`` that the application assigned to
retrieved chunks. Any ID in the answer that is not in that set is removed from
the displayed text and reported, so fabricated sources are never shown.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import BaseModel

from documind.models import Citation, RetrievedChunk

# Matches [S1], [S1, S3], [S1; S2], [s2] and the common "【S1】" variant.
_CITATION_GROUP = re.compile(r"[\[【]\s*([Ss]\d+(?:\s*[,;]\s*[Ss]\d+)*)\s*[\]】]")
_ID = re.compile(r"[Ss](\d+)")


class CitationCheck(BaseModel):
    text: str
    valid_ids: list[str]
    invalid_ids: list[str]


def build_citations(retrieved: Sequence[RetrievedChunk]) -> list[Citation]:
    return [
        Citation(source_id=f"S{i}", chunk=item.chunk, similarity=item.similarity)
        for i, item in enumerate(retrieved, start=1)
    ]


def validate_citations(text: str, sources: Sequence[Citation]) -> CitationCheck:
    known = {s.source_id for s in sources}
    valid: list[str] = []
    invalid: list[str] = []

    def replace(match: re.Match[str]) -> str:
        kept: list[str] = []
        for num in _ID.findall(match.group(1)):
            sid = f"S{int(num)}"
            if sid in known:
                kept.append(sid)
                if sid not in valid:
                    valid.append(sid)
            elif sid not in invalid:
                invalid.append(sid)
        return "".join(f"[{sid}]" for sid in kept)

    cleaned = _CITATION_GROUP.sub(replace, text)
    cleaned = re.sub(r"[ \t]+([.,;:])", r"\1", cleaned)  # tidy "word [S9]." -> "word."
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return CitationCheck(text=cleaned.strip(), valid_ids=valid, invalid_ids=invalid)
