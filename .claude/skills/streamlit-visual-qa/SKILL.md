---
name: streamlit-visual-qa
description: Visual and functional QA for the DocuMind Streamlit UI - responsive viewports, accessibility, keyboard focus, dark/light contrast, rerun stability, chat composer position, sidebar collapse, screenshot regression and empty/error states. Use after any UI change and before committing.
---

# Streamlit visual QA

Never sign off a UI change from source code alone. Run the app and look at it.

## 1. Start

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501 --server.headless true
```

Use the in-app browser (or Playwright) at <http://localhost:8501>. Note which checks could not be run and say so.

## 2. Viewports

Check each of: **1440×900**, **1366×768**, **1024×768**, **390×844** (mobile).

- No horizontal scrollbar (`document.documentElement.scrollWidth <= innerWidth`).
- Composer visible and usable; messages never hidden behind it.
- Sidebar collapses (chevron) and the main column re-centres; on mobile it starts collapsed.
- Long filenames wrap or truncate with ellipsis — never overflow.

## 3. States to capture (screenshot each)

1. Chat empty (welcome + suggestions) with and without documents.
2. One question + one answer with citations; `Sources` collapsed and expanded.
3. Long multi-paragraph / table / code answer.
4. 6+ messages: scroll, composer stays put, no duplicated messages after reruns.
5. Retrieval-only mode (no key): evidence-first layout, compact notice, nothing claims to be generated.
6. Generation error (e.g. invalid key): friendly inline error, app still usable.
7. No documents: empty states on Chat and Documents link to upload.
8. Scope selector: all / some / none selected; "none" blocks the question with guidance.
9. Documents list: upload, duplicate, needs-OCR, failed, delete confirm.
10. Evaluation: run, metrics, failure analysis, saved results.
11. Settings: status matches reality (never "connected" without a key).

## 4. Accessibility

- Tab through sidebar, suggestions, scope control, composer, source expanders: focus ring visible (accent, 2px).
- Every icon-only button has a label/tooltip (`help=`).
- Contrast ≥ 4.5:1 for body and metadata text, ≥ 3:1 for UI boundaries — check with the token table in
  `minimalist-chat-ui`. Verify both themes if light mode is enabled.
- `prefers-reduced-motion: reduce` disables transitions.

## 5. Rerun behaviour

- Click a suggestion → exactly one user + one assistant message.
- Change scope, switch pages, come back → scope and conversation preserved.
- New chat clears messages but keeps scope; Clear does the same.
- Console: no errors (`read_console_messages onlyErrors`); server log: no tracebacks or deprecation spam.

## 6. Regression

```powershell
scripts\test.ps1
```

Compare new screenshots with `assets/` before replacing them; replace README screenshots only with real captures.

## Report format

List each check as pass / fail / not run, with the viewport and a one-line note. Fix fails, re-run, then report.
