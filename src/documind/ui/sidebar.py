"""Sidebar: wordmark, new chat, navigation, recent conversations, model status."""

from __future__ import annotations

from html import escape

import streamlit as st

from documind.app_factory import DocuMindApp
from documind.ui import session
from documind.ui.theme import MARK_SVG, dot, html


def render(app: DocuMindApp, pages: dict) -> None:
    with st.sidebar:
        html(f'<div class="dm-wordmark"><span class="dm-mark">{MARK_SVG}</span>DocuMind</div>')
        if st.button(
            "New chat",
            icon=":material/edit_square:",
            width="stretch",
            help="Start a new conversation",
        ):
            session.new_conversation()
            st.switch_page(pages["chat"])

        st.page_link(pages["chat"], label="Chat", icon=":material/chat_bubble:")
        st.page_link(pages["documents"], label="Documents", icon=":material/description:")
        st.page_link(pages["evaluation"], label="Evaluation", icon=":material/monitoring:")

        recent = session.recent_conversations()
        if recent:
            html('<div class="dm-label">Recent</div>')
            with st.container(key="dm-recent", gap=None):
                for cid, title in recent:
                    current = cid == st.session_state.current_conv
                    if st.button(
                        f"**{title}**" if current else title,
                        key=f"conv_{cid}",
                        type="tertiary",
                        width="stretch",
                        help=title,
                    ):
                        st.session_state.current_conv = cid
                        st.switch_page(pages["chat"])

        st.space("large")
        status = session.model_status(app)
        html(
            f'<div class="dm-status" title="{escape(status.detail)}">{dot(status.kind)}'
            f"<span>{escape(status.label)}</span></div>"
        )
        st.page_link(pages["settings"], label="Settings", icon=":material/settings:")
        st.page_link(pages["about"], label="About", icon=":material/info:")
