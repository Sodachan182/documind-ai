"""Documents page: upload, list, inspect, re-index and delete."""

from __future__ import annotations

from html import escape

import streamlit as st

from documind.app_factory import DocuMindApp
from documind.models import DocumentRecord, DocumentStatus
from documind.ui import session
from documind.ui.theme import dot, empty_state, html, notice, page_header
from documind.utils.file_utils import human_size

_OUTCOME_TEXT = {
    "indexed": ("ok", "Indexed"),
    "duplicate": ("muted", "Already in the knowledge base"),
    "needs_ocr": ("warn", "Needs OCR"),
    "rejected": ("err", "Rejected"),
    "failed": ("err", "Failed"),
}


def ingest_files(app: DocuMindApp, files: list[tuple[str, bytes]]) -> None:
    """Index files with a single progress bar; keep a result summary for the next run."""
    bar = st.progress(0.0, text="Preparing…")
    results = []
    for i, (name, data) in enumerate(files):

        def report(p: float, msg: str, i: int = i, name: str = name) -> None:
            bar.progress(min((i + p) / len(files), 1.0), text=f"{name}: {msg}")

        result = app.ingestion.ingest(name, data, report)
        results.append({"name": result.filename, "outcome": result.outcome, "message": result.message})
    bar.empty()
    st.session_state.ingest_results = results


def load_sample_documents(app: DocuMindApp) -> None:
    ingest_files(app, [(p.name, p.read_bytes()) for p in sorted(session.DEMO_DIR.iterdir())])


def _status(record: DocumentRecord, needs_reindex: bool) -> tuple[str, str]:
    if record.status == DocumentStatus.INDEXED:
        return ("warn", "Needs re-index") if needs_reindex else ("ok", "Indexed")
    if record.status == DocumentStatus.NEEDS_OCR:
        return "warn", "Needs OCR"
    return "err", "Failed"


def _contents(record: DocumentRecord) -> str:
    if record.file_type.value == "pdf":
        unit = f"{record.page_count} page{'s' if record.page_count != 1 else ''}"
    else:
        unit = f"{record.paragraph_count} paragraphs"
    return f"{unit}, {record.chunk_count} chunks"


@st.dialog("Indexed chunks", width="large")
def _inspect(app: DocuMindApp, record: DocumentRecord) -> None:
    chunks = app.registry.load_chunks(record.document_id)
    st.caption(f"{record.filename}: {len(chunks)} chunks. This is exactly the text that is embedded and searched.")
    for chunk in chunks:
        st.markdown(f"**{chunk.chunk_index}.** {chunk.location_label}  \n:small[{len(chunk.content)} characters]")
        st.text(chunk.content)


@st.dialog("Delete document?")
def _confirm_delete(app: DocuMindApp, record: DocumentRecord) -> None:
    st.write(
        f"**{record.filename}** and its {record.chunk_count} indexed chunks will be removed from the knowledge "
        "base. This cannot be undone; you can upload the file again later."
    )
    c1, c2 = st.columns(2)
    if c1.button("Cancel", width="stretch"):
        st.rerun()
    if c2.button("Delete", type="primary", width="stretch", icon=":material/delete:"):
        app.ingestion.delete(record.document_id)
        st.toast(f"Deleted {record.filename}")
        st.rerun()


def _row(app: DocuMindApp, record: DocumentRecord) -> None:
    needs_reindex = app.ingestion.needs_reindex(record)
    kind, label = _status(record, needs_reindex)
    icon = ":material/picture_as_pdf:" if record.file_type.value == "pdf" else ":material/article:"
    with st.container(key=f"dmrow-{record.document_id}"):
        c = st.columns([5, 1.1, 2.4, 1.7, 0.7], vertical_alignment="center")
        with c[0]:
            st.markdown(f"{icon}&nbsp; {escape(record.filename)}")
        c[1].markdown(
            f'<span class="dm-cell">{"PDF" if record.file_type.value == "pdf" else "Word"}</span>',
            unsafe_allow_html=True,
        )
        c[2].markdown(f'<span class="dm-cell">{escape(_contents(record))}</span>', unsafe_allow_html=True)
        c[3].markdown(f'<span class="dm-cell-status">{dot(kind)}{escape(label)}</span>', unsafe_allow_html=True)
        with c[4].popover("", icon=":material/more_horiz:", type="tertiary", help=f"Actions for {record.filename}"):
            if st.button(
                "Inspect chunks",
                icon=":material/visibility:",
                type="tertiary",
                disabled=record.chunk_count == 0,
                key=f"insp_{record.document_id}",
            ):
                _inspect(app, record)
            if needs_reindex and st.button(
                "Re-index", icon=":material/refresh:", type="tertiary", key=f"reidx_{record.document_id}"
            ):
                with st.spinner("Re-indexing…"):
                    app.ingestion.reindex(record.document_id)
                st.rerun()
            if st.button("Delete…", icon=":material/delete:", type="tertiary", key=f"del_{record.document_id}"):
                _confirm_delete(app, record)
        detail = record.error if record.status != DocumentStatus.INDEXED else None
        detail = detail or (record.warnings[0] if record.warnings else None)
        if detail:
            html(f'<div class="dm-muted" style="margin:-0.35rem 0 0.4rem 1.9rem">{escape(detail)}</div>')


def render(app: DocuMindApp) -> None:
    records = app.ingestion.list_documents()
    total_chunks = sum(r.chunk_count for r in records)
    if records:
        subtitle = f"{len(records)} document{'s' if len(records) != 1 else ''}, {total_chunks} chunks indexed."
    else:
        subtitle = "PDF and Word files you add here are what DocuMind searches."
    page_header("Documents", subtitle)

    files = st.file_uploader(
        f"Add PDF or Word files (up to {app.settings.max_upload_mb} MB each)",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
    )
    with st.container(horizontal=True, gap="small"):
        if st.button(
            f"Index {len(files)} file{'s' if len(files) != 1 else ''}" if files else "Index files",
            type="primary",
            disabled=not files,
            icon=":material/upload:",
        ):
            ingest_files(app, [(f.name, f.getvalue()) for f in files])
            st.session_state.uploader_key += 1  # clear the uploader so files are not re-indexed
            st.rerun()
        if st.button(
            "Load sample documents",
            type="tertiary",
            icon=":material/folder_special:",
            help="Three fictional company documents for trying DocuMind",
        ):
            load_sample_documents(app)
            st.rerun()

    for item in st.session_state.pop("ingest_results", []):
        kind, label = _OUTCOME_TEXT[item["outcome"]]
        notice(f"{item['name']}: {label}. {item['message']}", kind)

    stale = [r for r in records if app.ingestion.needs_reindex(r)]
    if stale:
        notice(
            f"{len(stale)} document(s) were indexed with a different embedding model and are excluded from "
            "search until re-indexed.",
            "warn",
        )
        if st.button("Re-index all", icon=":material/refresh:"):
            for record in stale:
                with st.spinner(f"Re-indexing {record.filename}…"):
                    app.ingestion.reindex(record.document_id)
            st.rerun()

    if not records:
        empty_state("No documents yet", "Upload a PDF or Word file above, or load the sample documents.")
        return

    st.write("")
    head = st.columns([5, 1.1, 2.4, 1.7, 0.7])
    for col, text in zip(head, ["Name", "Type", "Contents", "Status", ""], strict=True):
        col.markdown(f'<span class="dm-th">{text}</span>', unsafe_allow_html=True)
    for record in records:
        _row(app, record)
    html(
        f'<p class="dm-muted" style="margin-top:1rem">Storage: {escape(human_size(_disk(app)))} on disk in '
        f"{escape(str(session.persist_dir()))}</p>"
    )


def _disk(app: DocuMindApp) -> int:
    root = app.store.persist_dir
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) if root.exists() else 0
