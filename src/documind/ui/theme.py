"""Design tokens and the small CSS layer for DocuMind-owned markup.

Native colours, fonts and radii come from ``.streamlit/config.toml``. The CSS
below only styles:
  * classes DocuMind emits itself (``dm-*``),
  * containers DocuMind creates with a ``key`` (Streamlit adds ``st-key-<key>``),
  * a few documented ``data-testid`` hooks for layout width.
Hashed ``st-emotion-*`` classes are never targeted.
"""

from __future__ import annotations

from html import escape

import streamlit as st

TOKENS: dict[str, dict[str, str]] = {
    "dark": {
        "bg": "#0f1013",
        "elevated": "#1c1e24",
        "hover": "rgba(255, 255, 255, 0.05)",
        "text": "#ececef",
        "text2": "#a4a7b0",
        "muted": "#8a8e99",
        "border": "#2a2c33",
        "accent": "#8b93f8",
        "success": "#6bbd9f",
        "warning": "#d9b26b",
        "danger": "#e08585",
    },
    "light": {
        "bg": "#fbfaf8",
        "elevated": "#f0efeb",
        "hover": "rgba(0, 0, 0, 0.045)",
        "text": "#1d1e22",
        "text2": "#4a4d55",
        "muted": "#666a72",
        "border": "#e2e0db",
        "accent": "#4f56d6",
        "success": "#2f7d61",
        "warning": "#8a610d",
        "danger": "#b23b3b",
    },
}


def theme_type() -> str:
    try:
        kind = st.context.theme.type
    except Exception:
        kind = None
    return kind if kind in TOKENS else "dark"


def css() -> str:
    return """
<style>
:root {
  /* Neutral tones derive from the inherited text colour, so they are correct in both
     light and dark themes without detecting which one is active. */
  --dm-text2: color-mix(in srgb, currentColor 74%, transparent);
  --dm-muted: color-mix(in srgb, currentColor 64%, transparent);
  --dm-elevated: color-mix(in srgb, currentColor 7%, transparent);
  --dm-hover: color-mix(in srgb, currentColor 6%, transparent);
  --dm-border: color-mix(in srgb, currentColor 13%, transparent);
  --dm-accent: #6f78ec;  /* one indigo that holds 3:1+ on both theme backgrounds */
  --dm-success: #4fa889; --dm-warning: #c9952f; --dm-danger: #d16262;
}
/* Layout: one centred reading column. */
[data-testid="stMainBlockContainer"] { max-width: 820px; padding-top: 2.25rem; padding-bottom: 2rem; }
[data-testid="stBottomBlockContainer"] { max-width: 820px; }

/* Accessibility */
a:focus-visible, button:focus-visible, summary:focus-visible, [role="tab"]:focus-visible, [role="option"]:focus-visible {
  outline: 2px solid var(--dm-accent) !important; outline-offset: 2px; }
/* No custom animations: Streamlit's own motion (e.g. the sidebar slide) is left untouched. */

/* Typography helpers */
.dm-title { font-size: 1.375rem; font-weight: 600; letter-spacing: -0.01em; margin: 0 0 0.25rem; color: inherit; }
.dm-sub { color: var(--dm-text2); font-size: 0.9375rem; margin: 0 0 1.5rem; }
.dm-label { color: var(--dm-muted); font-size: 0.8125rem; font-weight: 500; margin: 1.25rem 0 0.35rem; }
.dm-muted { color: var(--dm-muted); font-size: 0.8125rem; line-height: 1.5; }

/* Sidebar */
.dm-wordmark { display: flex; align-items: center; gap: 0.55rem; font-weight: 600; font-size: 1.02rem;
  color: inherit; padding: 0 0.25rem 0.75rem; letter-spacing: -0.01em; }
.dm-mark { width: 22px; height: 22px; border-radius: 6px; background: var(--dm-accent); display: grid; place-items: center; }
.dm-mark svg { width: 13px; height: 13px; }
.dm-status { display: flex; align-items: center; gap: 0.5rem; font-size: 0.8125rem; color: var(--dm-text2);
  padding: 0.25rem 0.25rem; }
.dm-dot { width: 7px; height: 7px; border-radius: 50%; flex: none; background: var(--dm-muted); }
.dm-dot.ok { background: var(--dm-success); } .dm-dot.warn { background: var(--dm-warning); }
.dm-dot.err { background: var(--dm-danger); } .dm-dot.accent { background: var(--dm-accent); }
.st-key-dm-recent button, .st-key-dm-suggestions button { justify-content: flex-start; text-align: left; }
.st-key-dm-recent button p { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 200px;
  color: var(--dm-text2); font-size: 0.875rem; }

/* Welcome */
.dm-welcome { text-align: center; margin: 14vh 0 2rem; }
.dm-welcome h1 { font-size: 1.625rem; font-weight: 600; letter-spacing: -0.015em; margin: 0 0 0.4rem; padding: 0;
  color: inherit; }
.dm-welcome p { color: var(--dm-text2); font-size: 0.975rem; margin: 0; }
.st-key-dm-suggestions { max-width: 560px; margin: 0 auto; }
.st-key-dm-suggestions button { padding: 0.6rem 0.75rem; border-radius: 10px; color: var(--dm-text2); width: 100%; }
.st-key-dm-suggestions button:hover { background: var(--dm-hover); color: inherit; }

/* Messages */
.dm-user { display: flex; justify-content: flex-end; margin: 1.5rem 0 1rem; }
.dm-bubble { background: var(--dm-elevated); color: inherit; padding: 0.65rem 0.95rem; border-radius: 14px;
  max-width: 78%; white-space: pre-wrap; overflow-wrap: anywhere; font-size: 0.975rem; line-height: 1.55; }
[class*="st-key-dm-answer-"] { font-size: 0.975rem; line-height: 1.65; }
[class*="st-key-dm-answer-"] table { font-size: 0.9rem; }
.dm-notice { display: flex; gap: 0.55rem; align-items: baseline; color: var(--dm-text2); font-size: 0.9rem;
  line-height: 1.55; margin: 0.15rem 0 0.6rem; }
.dm-notice .dm-dot { transform: translateY(-1px); }
.dm-meta { color: var(--dm-muted); font-size: 0.78rem; margin-top: 0.5rem; }

/* Sources */
.dm-source { padding: 0.7rem 0 0.8rem; border-top: 1px solid var(--dm-border); }
.dm-source:first-child { border-top: none; padding-top: 0.2rem; }
.dm-source-head { display: flex; align-items: baseline; gap: 0.5rem; flex-wrap: wrap; font-size: 0.875rem; }
.dm-num { min-width: 1.35rem; height: 1.35rem; border-radius: 6px; display: inline-grid; place-items: center;
  font-size: 0.75rem; font-weight: 600; color: inherit; border: 1px solid var(--dm-border); }
.dm-file { font-weight: 600; color: inherit; overflow-wrap: anywhere; }
.dm-loc { color: var(--dm-text2); }
.dm-quote { opacity: 1; margin: 0.45rem 0 0.3rem 1.85rem; padding: 0 0 0 0.75rem; border-left: 2px solid var(--dm-border);
  color: var(--dm-text2); font-size: 0.875rem; line-height: 1.55; white-space: pre-wrap; overflow-wrap: anywhere; }
.dm-source .dm-muted { margin-left: 1.85rem; font-size: 0.75rem; }
.dm-group { color: var(--dm-muted); font-size: 0.78rem; font-weight: 500; margin: 0.9rem 0 0.2rem; }

/* Documents list */
.dm-th { color: var(--dm-muted); font-size: 0.78rem; font-weight: 500; }
[class*="st-key-dmrow-"] { border-top: 1px solid var(--dm-border); padding-top: 0.45rem; }
.dm-docname { font-weight: 500; color: inherit; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.dm-cell { color: var(--dm-text2); font-size: 0.875rem; }
.dm-cell-status { display: flex; align-items: center; gap: 0.45rem; color: var(--dm-text2); font-size: 0.875rem; }

/* Key-value lists (Settings) */
.dm-kv { display: grid; grid-template-columns: minmax(140px, 34%) 1fr; gap: 0.5rem 1rem; font-size: 0.9rem;
  margin: 0.25rem 0 0.5rem; }
.dm-kv dt { color: var(--dm-muted); } .dm-kv dd { margin: 0; color: inherit; overflow-wrap: anywhere; }
.dm-empty { text-align: center; padding: 3rem 1rem; color: var(--dm-text2); }
.dm-empty h3 { font-size: 1.05rem; font-weight: 600; color: inherit; margin: 0 0 0.35rem; padding: 0; }
</style>
"""


def inject() -> None:
    st.html(css())


def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def dot(kind: str) -> str:
    return f'<span class="dm-dot {escape(kind)}" aria-hidden="true"></span>'


def page_header(title: str, subtitle: str = "") -> None:
    sub = f'<p class="dm-sub">{escape(subtitle)}</p>' if subtitle else ""
    html(f'<h1 class="dm-title">{escape(title)}</h1>{sub}')


def notice(text: str, kind: str = "muted") -> None:
    html(f'<div class="dm-notice" role="status">{dot(kind)}<span>{escape(text)}</span></div>')


def empty_state(title: str, body: str) -> None:
    html(f'<div class="dm-empty"><h3>{escape(title)}</h3><p>{escape(body)}</p></div>')


MARK_SVG = (
    '<svg viewBox="0 0 16 16" fill="none" aria-hidden="true"><path d="M4 3.5h5.2L12 6.3v6.2H4z" '
    'stroke="white" stroke-width="1.5" stroke-linejoin="round"/><path d="M6.3 8.6h3.4M6.3 10.6h2.2" '
    'stroke="white" stroke-width="1.4" stroke-linecap="round"/></svg>'
)
