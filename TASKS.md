# TASKS

Progress log for DocuMind AI. `[x]` = done and verified, `[ ]` = open, `[~]` = done with a caveat.

## v1.0 — RAG application

- [x] Environment inspection (Python 3.12 venv, Git, GitHub credential via Git Credential Manager)
- [x] PDF extraction with exact page numbers; scanned-PDF (OCR-needed) detection
- [x] DOCX extraction: paragraphs + tables in body order, section headings, paragraph indexes
- [x] Unicode normalisation (NFC, ligatures, hyphenation, control characters)
- [x] Page/section-bounded chunking with overlap and stable IDs
- [x] Upload validation: size, extension + signature, zip-bomb guard, filename sanitisation, SHA-256 duplicates
- [x] Gemini embeddings (batched, retried, dimension-checked) + offline lexical embedder
- [x] ChromaDB store with explicit vectors, per-model collections, incompatible-space detection, re-index
- [x] Retriever with document-ID filtering, Python scope re-check, de-duplication, context budget
- [x] Grounded prompt with untrusted-source fencing and escaping; structured JSON output; abstention flag
- [x] App-controlled citation IDs; fabricated citations stripped and reported
- [x] Synthetic demo documents (reproducible) + 20 labelled questions (EN/VI, conflicts, unanswerable, isolation)
- [x] Evaluation: Recall@k, Hit@k, MRR, source identification, scope leaks; optional answer-level checks
- [x] Offline evaluation run and recorded (`docs/evaluation_results/offline.json`)
- [x] Windows launcher (`START_DOCUMIND.bat`, `scripts/*.ps1`) — fresh install verified in a path with spaces
- [x] Pytest suite (offline) + Ruff; GitHub Actions on Ubuntu and Windows
- [x] Documentation: README, architecture, evaluation, security, deployment, demo script
- [~] Live Gemini path — implemented and unit-tested with SDK fakes; **not run against the real API** (no key available)
- [ ] Gemini-embedding / answer-level evaluation (`demo/run_evaluation.py --mode gemini`) — needs an API key

## v1.1 — UI redesign (minimalist conversational workspace)

Design rules: `.claude/skills/minimalist-chat-ui/SKILL.md`. QA procedure: `.claude/skills/streamlit-visual-qa/SKILL.md`.

- [~] Official `frontend-design` plugin: offered for install from the plugin catalog (install is a user action);
      its published SKILL.md guidance was read and applied for this redesign
- [x] Project skills `minimalist-chat-ui` and `streamlit-visual-qa`
- [x] Remove gradient hero, emoji navigation, badge clusters and warning panels
- [x] Native theme for dark (primary) and light, Inter font, one indigo accent; CSS limited to `dm-*` classes,
      keyed containers and two layout `data-testid` hooks
- [x] Sidebar: wordmark, New chat, Chat / Documents / Evaluation, recent conversations, compact model status,
      Settings, About (hidden `st.navigation` + `st.page_link`)
- [x] Chat: centred time-of-day welcome, 4 scope-aware suggestions, right-aligned user bubbles, unboxed answers
- [x] Composer pinned with `st.bottom` + `st.chat_input`; document chips (`st.pills`) above it; popover with search
      for more than 6 documents; disabled with guidance when nothing is selected
- [x] Citations: inline numbered badges, collapsed `Sources (n)`, verbatim passages, cited-first numbering
- [x] Retrieval-only mode: compact notice + evidence-first layout (top 3 inline, rest folded)
- [x] Documents: table-style list, status dots, actions menu, inspect and delete-confirm dialogs
- [x] Evaluation: quiet metrics, questions table, failure cases, saved results
- [x] Settings: verified model status (free `models.get` check), session-only key handling, retrieval and storage info
- [x] Model status never says "connected" without a successful check or answer
- [x] Visual QA at 1440×900, 1366×768, 1024×768, 390×844; dark and light; keyboard focus; no horizontal overflow
- [x] Regression: 102 offline tests, Ruff lint + format
- [x] README screenshots replaced with real captures

## v1.2 — Multi-provider generation (ShopAIKey)

- [x] Provider interface (`llm/base.py`) shared by Gemini direct and three ShopAIKey adapters
- [x] ShopAIKey OpenAI-compatible (`openai` SDK, base `/v1`), Anthropic-compatible (`anthropic` SDK, root),
      Gemini-compatible (`google-genai`, `http_options.base_url`)
- [x] Independent embedding configuration; optional ShopAIKey embeddings with explicit model and dimension
- [x] Config reconciled with existing names (legacy aliases kept); gateway URL safety checks
- [x] Model catalog per protocol (on click), manual model ID, session persistence
- [x] Error states: not configured, ready, connected, auth failed, insufficient balance, model unavailable,
      rate limited, service error; bounded retries; one request per submission
- [x] Settings UI (provider, protocol, key, model, test), sidebar active-model indicator, third-party disclosure
- [x] Conversation history (last 2 exchanges) passed to every provider; retrieval still per question
- [x] 53 new offline tests (fake SDK clients); 155 total passing; Ruff clean
- [ ] **Live ShopAIKey verification**: no key was available. Run `pytest -m live tests/test_live_shopaikey.py`
      with `SHOPAIKEY_API_KEY` and `SHOPAIKEY_MODEL` set, or use Settings → Test connection
- [ ] ShopAIKey embeddings: implemented and mock-tested, not verified against a live key; off by default
