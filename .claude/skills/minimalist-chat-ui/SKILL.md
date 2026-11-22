---
name: minimalist-chat-ui
description: Design and refine minimalist AI chat interfaces with strong typography, restrained visual hierarchy, accessible interactions, and anti-AI-slop design principles. Use for DocuMind interface work (Streamlit UI in app.py and src/documind/ui/).
---

# Minimalist chat UI

Use this skill whenever you change how DocuMind looks or behaves in the browser. The goal is a mature,
quiet product: the conversation and the evidence are the interface; everything else gets out of the way.

## Philosophy (in priority order)

1. Content first, conversation first.
2. Typography before decoration; whitespace before borders.
3. Clarity before novelty; functionality before animation.
4. Consistency before experimentation.
5. Neutral surfaces before gradients; subtle feedback before effects.

If a visual element does not encode information or afford an action, remove it.

## Design tokens (single source: `src/documind/ui/theme.py`)

| Token | Dark | Light | Use |
|---|---|---|---|
| `bg` | `#0f1013` | `#fbfaf8` | app background |
| `sidebar` | `#15161a` | `#f3f2ef` | sidebar |
| `elevated` | `#1c1e24` | `#ffffff` | user bubble, menus, composer |
| `hover` | `#25272e` | `#ebe9e4` | hover rows |
| `text` | `#ececef` | `#1d1e22` | primary text |
| `text-2` | `#a4a7b0` | `#55585f` | secondary |
| `muted` | `#7c808b` | `#6f7279` | metadata (≥ 4.5:1 on bg) |
| `border` | `#2a2c33` | `#e2e0db` | hairlines only |
| `accent` | `#8b93f8` | `#4f56d6` | the only accent: focus, links, selected nav, citations |
| `success/warning/danger` | `#6bbd9f` / `#d9b26b` / `#dc8080` | `#2f7d61` / `#94690f` / `#b23b3b` | status dots and inline text, never fills |

Radii: 8 (controls), 12 (bubbles, composer), 16 (rare, large surfaces). Spacing: multiples of 4.

## Typography

- Inter (Google Fonts) with `system-ui, -apple-system, "Segoe UI", Roboto, sans-serif` fallback.
- Body 15px / 1.6. Assistant answers 15.5px / 1.65, max line length ~75ch (conversation column 760px).
- Page titles 20px/600; section labels 13px/600 sentence case. Metadata 12.5–13px in `muted`.
- Never all-caps labels, eyebrows above titles, or one accented word in a heading.

## Layout

- Sidebar ~248px: wordmark, New chat, nav (Chat, Documents, Evaluation), then a bottom block with
  model status and Settings/About. No large widgets, no warning panels.
- Main column centered, max 760px for chat, ~960px for tables.
- Empty chat: centered greeting (time-of-day) + one-line helper + 4 quiet suggestion rows.
- Chat: user bubble right-aligned on `elevated`, max ~80% width; assistant text unboxed, full column width.
- Composer: native `st.chat_input` (pinned to the bottom by Streamlit); scope selector directly above it.

## Components

- **Citations**: inline `[1]` markers in the answer (accent colour); below, one collapsed `Sources · n` expander
  (here the middle dot separates a label from a count — fine), listing filename, location and passage.
  Never show passages expanded by default except in retrieval-only mode, where evidence *is* the answer.
- **Status**: a 6px dot + short text ("Gemini connected", "Offline retrieval"). Details live in Settings.
- **Documents**: a list/table — name, type, size of structure, status dot, actions. No card grid.
- **Metrics**: plain numbers with labels, no coloured tiles, at most one row.

## Anti-AI-slop checklist (reject any of these)

Giant gradient heroes · purple glows · nested cards · emoji navigation · badge clusters · uppercase eyebrows ·
marketing copy ("Unlock the power of…") · oversized type · more than one accent · decorative progress bars ·
heavy shadows · gratuitous animation · floating decorations · noisy metric dashboards · borders on every row ·
an icon beside every label · trailing arrows on links · middle-dot metadata chains.

## Streamlit implementation rules

- Prefer native APIs: `st.chat_message`, `st.chat_input`, `st.popover`, `st.expander`, `st.dataframe`,
  `st.navigation`-free radio/buttons with `icon=":material/...:"` (Material Symbols ship with Streamlit).
- Style through `.streamlit/config.toml` theme options first; inject CSS only for DocuMind-owned classes
  (`dm-*`) and for stable `data-testid` hooks when no option exists. Never target hashed `st-emotion-*` classes.
- Escape every user- or document-derived string inserted into HTML (`html.escape`).
- Keep state in `st.session_state` with durable keys separate from widget keys (widgets on hidden pages are
  garbage-collected).
- Respect `prefers-reduced-motion`; keep transitions ≤ 150ms and only on hover/focus.

## Process

1. Write/adjust tokens before components. 2. Build. 3. Run the app and screenshot (see the
`streamlit-visual-qa` skill). 4. Critique against the checklist above; cut anything decorative. 5. Repeat.
