"""JSON document registry + per-document chunk snapshots.

Chunk snapshots let documents be re-embedded after an embedding-model change
without needing the original upload (which is never stored).
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from documind.models import Chunk, DocumentRecord

_LOCK = threading.RLock()


def _atomic_write(path: Path, payload: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, path)


class DocumentRegistry:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.chunks_dir = self.root / "chunks"
        self.chunks_dir.mkdir(exist_ok=True)
        self.path = self.root / "registry.json"

    def _load(self) -> dict[str, DocumentRecord]:
        if not self.path.exists():
            return {}
        raw = json.loads(self.path.read_text(encoding="utf-8") or "{}")
        return {k: DocumentRecord.model_validate(v) for k, v in raw.items()}

    def _save(self, records: dict[str, DocumentRecord]) -> None:
        payload = {k: v.model_dump(mode="json") for k, v in records.items()}
        _atomic_write(self.path, json.dumps(payload, ensure_ascii=False, indent=2))

    def all(self) -> list[DocumentRecord]:
        with _LOCK:
            return sorted(self._load().values(), key=lambda r: r.created_at)

    def get(self, document_id: str) -> DocumentRecord | None:
        with _LOCK:
            return self._load().get(document_id)

    def find_by_hash(self, file_hash: str) -> DocumentRecord | None:
        with _LOCK:
            return next((r for r in self._load().values() if r.file_hash == file_hash), None)

    def upsert(self, record: DocumentRecord) -> None:
        with _LOCK:
            records = self._load()
            records[record.document_id] = record
            self._save(records)

    def remove(self, document_id: str) -> None:
        with _LOCK:
            records = self._load()
            records.pop(document_id, None)
            self._save(records)
            self._chunk_path(document_id).unlink(missing_ok=True)

    # -- chunk snapshots --------------------------------------------------------
    def _chunk_path(self, document_id: str) -> Path:
        safe = "".join(ch for ch in document_id if ch.isalnum() or ch in "_-")
        return self.chunks_dir / f"{safe}.json"

    def save_chunks(self, document_id: str, chunks: list[Chunk]) -> None:
        payload = json.dumps([c.model_dump(mode="json") for c in chunks], ensure_ascii=False)
        _atomic_write(self._chunk_path(document_id), payload)

    def load_chunks(self, document_id: str) -> list[Chunk]:
        path = self._chunk_path(document_id)
        if not path.exists():
            return []
        return [Chunk.model_validate(c) for c in json.loads(path.read_text(encoding="utf-8"))]
