"""DocuMind AI — Streamlit entry point.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st

from documind.ui import chat, documents, evaluation, session, settings, sidebar, theme

st.set_page_config(
    page_title="DocuMind",
    page_icon=":material/description:",
    layout="centered",
    initial_sidebar_state=260,  # native "auto" behaviour (smooth collapse, hidden on phones) at 260px
)
theme.inject()
session.init_state()
app = session.get_app()

pages = {
    "chat": st.Page(
        lambda: chat.render(app, pages), title="Chat", icon=":material/chat_bubble:", url_path="chat", default=True
    ),
    "documents": st.Page(
        lambda: documents.render(app), title="Documents", icon=":material/description:", url_path="documents"
    ),
    "evaluation": st.Page(
        lambda: evaluation.render(app), title="Evaluation", icon=":material/monitoring:", url_path="evaluation"
    ),
    "settings": st.Page(
        lambda: settings.render_settings(app), title="Settings", icon=":material/settings:", url_path="settings"
    ),
    "about": st.Page(lambda: settings.render_about(app), title="About", icon=":material/info:", url_path="about"),
}

current = st.navigation(list(pages.values()), position="hidden")
sidebar.render(app, pages)
current.run()
