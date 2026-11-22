"""Chat page: welcome state, conversation, document scope and composer."""

from __future__ import annotations

import contextlib
from datetime import datetime
from html import escape
from zoneinfo import ZoneInfo

import streamlit as st

from documind.app_factory import DocuMindApp
from documind.models import RAGAnswer
from documind.services.rag_service import QuestionValidationError
from documind.ui import session
from documind.ui.citations import render_answer
from documind.ui.documents import load_sample_documents
from documind.ui.theme import html, notice

MAX_PILLS = 6


def greeting() -> str:
    now = datetime.now()
    with contextlib.suppress(Exception):  # unknown or missing zone: fall back to server time
        if st.context.timezone:
            now = datetime.now(ZoneInfo(st.context.timezone))
    if 5 <= now.hour < 12:
        return "Good morning"
    if 12 <= now.hour < 18:
        return "Good afternoon"
    return "Good evening"


def suggestions(names: list[str]) -> list[tuple[str, str, str]]:
    """(label, prompt, icon) adapted to the documents currently in scope."""
    if not names:
        return []
    a = names[0]
    items = [(f"Summarize {a}", f"Summarize the key points of {a}.", ":material/notes:")]
    if len(names) >= 2:
        b = names[1]
        items.append(
            (f"Compare {a} and {b}", f"Compare {a} and {b}. Where do they differ?", ":material/compare_arrows:")
        )
    else:
        items.append(
            (f"List the main rules in {a}", f"What are the main rules and requirements in {a}?", ":material/checklist:")
        )
    items.append(
        (
            "Find deadlines and time limits",
            "What deadlines, time limits and retention periods do the documents specify?",
            ":material/search:",
        )
    )
    if len(names) >= 2:
        items.append(
            (
                "Identify conflicting information",
                "Identify any conflicting or inconsistent information between the selected documents.",
                ":material/rule:",
            )
        )
    else:
        items.append(("Find specific numbers", f"Which numeric limits and thresholds appear in {a}?", ":material/tag:"))
    return items


def _welcome(title: str, body: str) -> None:
    html(f'<div class="dm-welcome"><h1>{escape(title)}</h1><p>{escape(body)}</p></div>')


def _scope_bar(app: DocuMindApp, searchable: list[str], names: dict[str, str], pages: dict) -> None:
    def from_pills() -> None:
        st.session_state.doc_scope = list(st.session_state.scope_pills)

    with st.container(horizontal=True, vertical_alignment="center", gap="small", wrap=True, key="dm-scope"):
        if len(searchable) <= MAX_PILLS:
            st.session_state.scope_pills = list(st.session_state.doc_scope)
            st.pills(
                "Documents to search",
                options=searchable,
                selection_mode="multi",
                format_func=lambda d: session.display_name(names.get(d, d), 26),
                key="scope_pills",
                on_change=from_pills,
                label_visibility="collapsed",
                help="Only the selected documents are searched.",
                width="stretch",
                wrap=True,  # chips wrap on narrow screens instead of scrolling out of view
            )
        else:
            scope = st.session_state.doc_scope
            with st.popover(f"{len(scope)} of {len(searchable)} documents", icon=":material/folder_open:"):
                query = st.text_input("Search documents", placeholder="Filter by name", key="scope_filter")
                c1, c2 = st.columns(2)
                if c1.button("Select all", width="stretch"):
                    st.session_state.doc_scope = list(searchable)
                    st.rerun()
                if c2.button("Select none", width="stretch"):
                    st.session_state.doc_scope = []
                    st.rerun()
                for doc_id in searchable:
                    if query and query.lower() not in names.get(doc_id, "").lower():
                        continue
                    checked = st.checkbox(names.get(doc_id, doc_id), value=doc_id in scope, key=f"scope_cb_{doc_id}")
                    if checked and doc_id not in scope:
                        scope.append(doc_id)
                    elif not checked and doc_id in scope:
                        scope.remove(doc_id)
        st.page_link(pages["documents"], label="Add documents", icon=":material/add:")


def _process(app: DocuMindApp, conv: dict, prompt: str) -> None:
    # One submission = at most one provider request, even if Streamlit reruns mid-flight.
    submission = (st.session_state.current_conv, len(conv["messages"]), prompt)
    if submission in st.session_state.processed:
        return
    st.session_state.processed.add(submission)
    scope = list(st.session_state.doc_scope)
    html(f'<div class="dm-user"><div class="dm-bubble">{escape(prompt)}</div></div>')
    n = len(scope)
    try:
        with st.spinner(f"Searching {n} document{'s' if n != 1 else ''}…"):
            answer = app.rag.ask(
                prompt, scope, st.session_state.top_k, history=session.history_for_model(conv["messages"])
            )
    except QuestionValidationError as exc:
        notice(str(exc), "warn")
        return
    render_answer(answer, key=str(len(conv["messages"]) + 1))
    status_changed = session.note_answer(answer)
    first = not conv["messages"]
    conv["messages"].append({"role": "user", "content": prompt})
    conv["messages"].append({"role": "assistant", "answer": answer.model_dump(mode="json")})
    if first:
        conv["title"] = prompt if len(prompt) <= 60 else prompt[:59].rstrip() + "…"
    if first or status_changed:
        st.rerun()  # refresh the sidebar (recent conversations, model status)


def render(app: DocuMindApp, pages: dict) -> None:
    searchable = session.sync_scope(app)
    names = {r.document_id: r.filename for r in app.ingestion.list_documents()}
    conv = session.current_conversation()
    messages = conv["messages"]
    pending = st.session_state.pop("pending_prompt", None)

    if not searchable:
        _welcome(f"{greeting()}.", "Add a PDF or Word document and DocuMind will answer questions from it.")
        with st.container(horizontal=True, horizontal_alignment="center", vertical_alignment="center", gap="small"):
            st.page_link(pages["documents"], label="Upload documents", icon=":material/upload:")
            if st.button("Load sample documents", type="tertiary", icon=":material/folder_special:"):
                load_sample_documents(app)
                st.rerun()
        with st.bottom:
            st.chat_input("Add documents to start asking questions", disabled=True, key="composer_off")
        return

    if not messages and not pending:
        scope_names = [session.display_name(names[d], 40) for d in st.session_state.doc_scope if d in names]
        _welcome(
            f"{greeting()}. What would you like to explore?",
            "Ask questions, compare documents, or check what your knowledge base says.",
        )
        with st.container(key="dm-suggestions", gap=None):
            for i, (label, prompt, icon) in enumerate(suggestions(scope_names)):
                if st.button(label, key=f"suggest_{i}", type="tertiary", icon=icon, width="stretch"):
                    st.session_state.pending_prompt = prompt
                    st.rerun()
            if not scope_names:
                notice("Select at least one document below to ask a question.", "warn")

    for i, message in enumerate(messages):
        if message["role"] == "user":
            html(f'<div class="dm-user"><div class="dm-bubble">{escape(message["content"])}</div></div>')
        else:
            render_answer(RAGAnswer.model_validate(message["answer"]), key=str(i))

    with st.bottom:
        _scope_bar(app, searchable, names, pages)
        has_scope = bool(st.session_state.doc_scope)
        prompt = st.chat_input(
            "Ask anything about your documents…" if has_scope else "Select a document above to ask a question",
            max_chars=app.settings.max_question_chars,
            submit_mode="disable",
            disabled=not has_scope,
            key="composer",
        )

    prompt = prompt or pending
    if prompt and st.session_state.doc_scope:
        _process(app, conv, prompt)
