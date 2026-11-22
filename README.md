<div align="center">

# DocuMind AI

**Turn documents into reliable answers.**

Upload PDF and Word documents, ask questions in English or Vietnamese, and get answers grounded in your files —
with every claim linked to the exact page, section or paragraph it came from.

[![CI](https://github.com/Sodachan182/documind-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/Sodachan182/documind-ai/actions/workflows/ci.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

</div>

---

## Why

Teams keep their rules in long PDFs and Word files — handbooks, specifications, security policies. Finding the
right clause is slow, and asking a general-purpose chatbot is risky: it may answer from its training data,
blend documents together, or invent a convincing but non-existent rule.

DocuMind is a **Retrieval-Augmented Generation (RAG)** application that answers *only* from the documents you
select and shows the evidence for every answer, so a person can verify it in seconds.

## What it does

| Capability | How |
|---|---|
| **Document ingestion** | PDF (PyMuPDF) and DOCX (python-docx, including tables), multi-file upload, size/type/signature validation, SHA-256 duplicate detection, scanned-PDF (OCR-needed) detection |
| **Citation-preserving chunking** | ~900-character paragraph-aware chunks with ~120-character overlap that **never cross a PDF page or DOCX section**, so each chunk maps to one exact location |
| **Semantic search** | Gemini embeddings (`gemini-embedding-001`, 768-d) stored in a persistent ChromaDB collection with explicit vectors and cosine distance |
| **Grounded answers** | Gemini directly or the ShopAIKey gateway (OpenAI-, Anthropic- or Gemini-compatible), with JSON output, strict grounding rules, explicit *insufficient-evidence* handling and answer language matched to the question (EN/VI) |
| **Verifiable citations** | The app assigns IDs `S1…Sn` to retrieved chunks; the answer's citations are validated and any ID the model invents is stripped and reported |
| **Multi-document analysis** | Select which documents are searched; comparison and conflict questions cite each document separately; unselected documents are never retrieved |
| **Evaluation** | 20-question labelled set with Recall@k, Hit@k, MRR, source identification, isolation leaks, and optional answer-level checks |
| **Safety** | Prompt-injection fencing of document text, path-traversal-safe filenames, zip-bomb guard, in-memory processing, secret redaction in logs |
| **Works without a key** | With no API key the full ingestion → ChromaDB → retrieval path still runs using an offline lexical embedder, clearly labelled *retrieval only* |

## Interface

A quiet, conversation-first workspace: the question, the answer and its evidence are the interface.

![Welcome screen](assets/chat-welcome.jpg)

| Chat in retrieval-only mode (no API key) | Light theme |
|---|---|
| ![Retrieval-only answer with numbered evidence](assets/chat-retrieval-only.png) | ![Light theme](assets/chat-light.jpg) |
| **Documents** | **Evaluation** |
| ![Documents list](assets/documents.jpg) | ![Evaluation page](assets/evaluation.jpg) |

All screenshots are real captures of this app running locally without a Gemini key. The one below shows the
**layout** of a generated answer. It was rendered by the real UI, but the answer text came from a scripted
test double because no Gemini key was available. It is not Gemini output. Its citations point at the passages
that were actually retrieved.

<img src="assets/answer-layout-preview.png" alt="Generated-answer layout preview (scripted test double, not Gemini output)" width="460">

**How the chat works**

- **Welcome**: a time-of-day greeting and four suggestions that adapt to the documents in scope
  (summarise, compare, find deadlines, identify conflicts).
- **Scope chips** above the composer show which documents are searched. Click one to include or exclude it.
  With more than six documents they collapse into a searchable picker. With nothing selected, the composer is
  disabled and explains why.
- **Answers** are plain text, not boxed. Claims end with numbered citation badges. **Sources (n)** expands to
  the file, page or section, and the verbatim passage. Retrieved-but-uncited passages are listed separately.
- **Retrieval-only mode** (no key) says so in one line and shows the strongest passages directly. Nothing is
  presented as a generated answer.
- **Sidebar**: New chat, Chat / Documents / Evaluation, recent conversations in this session, a one-line model
  status (*Offline retrieval*, *Gemini configured*, *Gemini connected*, *Gemini error*), Settings and About.
  The status only says *connected* after a verified check or a successful answer.
- **Theme**: dark by default on dark-mode systems, with a matching light theme. Streamlit follows the OS
  preference; switch it in the ⋮ menu under Settings, then Theme.

Design rules and the QA checklist used for this UI live in [`.claude/skills/`](.claude/skills) (see
[Project skills](#project-skills)).

## Architecture

```mermaid
flowchart LR
    subgraph Ingestion
        U[Upload PDF / DOCX] --> V[Validate<br/>size · extension · signature · SHA-256]
        V --> X[Extract<br/>PyMuPDF · python-docx]
        X --> N[Normalize Unicode<br/>NFC · ligatures · hyphenation]
        N --> C[Chunk<br/>page / section aware]
        C --> E[Embed<br/>Gemini RETRIEVAL_DOCUMENT]
        E --> DB[(ChromaDB<br/>vectors + metadata)]
        C --> R[(Registry + chunk snapshots)]
    end
    subgraph Question answering
        Q[Question] --> QV[Validate + detect language]
        QV --> QE[Embed<br/>RETRIEVAL_QUERY]
        QE --> S[Top-k search<br/>filtered to selected docs]
        DB -.-> S
        S --> F[Context filter<br/>scope re-check · dedupe · budget]
        F --> P[Grounded prompt<br/>sources fenced as untrusted data]
        P --> G[Gemini<br/>JSON answer + abstain flag]
        G --> CV[Citation validation<br/>strip unknown IDs]
        CV --> A[Answer + source evidence]
    end
```

The pipeline is plain, typed Python — no LangChain or LlamaIndex — so every step can be read, tested and
explained. Key modules:

```text
src/documind/
├── config.py                  # env/.env settings (Pydantic); API key excluded from repr
├── models.py                  # Chunk, DocumentRecord, Citation, RAGAnswer ...
├── document_processing/       # pdf_reader, docx_reader, cleaner, chunker
├── embeddings/                # Embedder ABC; Gemini, ShopAIKey (opt-in) and offline lexical embedders
├── retrieval/                 # VectorStore (ChromaDB), Retriever
├── llm/                       # provider interface, Gemini + ShopAIKey adapters, prompts, error mapping
├── citations/                 # app-controlled IDs + validation
├── services/                  # ingestion, RAG, evaluation, registry
├── ui/                        # Streamlit UI: theme, session, sidebar, chat, citations, documents, evaluation, settings
└── app_factory.py             # composition root
```

More detail: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Quick start (Windows)

**One click:** double-click **`START_DOCUMIND.bat`**.

It finds Python 3.12+, creates `.venv`, installs dependencies (only when `requirements.txt` changes), creates
`.env` from `.env.example`, validates the configuration and opens the app at <http://localhost:8501>. It needs
no administrator rights and works in folders with spaces.

**Manual fallback:**

```powershell
git clone https://github.com/Sodachan182/documind-ai.git
cd documind-ai
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env      # then add your GEMINI_API_KEY
.\.venv\Scripts\python.exe -m streamlit run app.py
```

**macOS / Linux:**

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

## AI providers

Answer generation and embeddings are configured independently ([docs/PROVIDERS.md](docs/PROVIDERS.md)):

| Answer generation (`DOCUMIND_LLM_PROVIDER`) | How |
|---|---|
| **Google Gemini** (direct) | `google-genai` SDK, `GEMINI_API_KEY`, `GEMINI_MODEL` |
| **ShopAIKey, OpenAI-compatible** | `openai` SDK, base `https://api.shopaikey.com/v1` |
| **ShopAIKey, Anthropic-compatible** | `anthropic` SDK, base `https://api.shopaikey.com` |
| **ShopAIKey, Gemini-compatible** | `google-genai` SDK with `http_options.base_url` |
| **Offline** | no model; chat shows retrieved evidence only |

| Embeddings (`DOCUMIND_EMBEDDING_PROVIDER`) | |
|---|---|
| **Gemini** | `gemini-embedding-001`, 768-d (default when `GEMINI_API_KEY` is set) |
| **ShopAIKey** (opt-in) | OpenAI-format `/v1/embeddings`; model and dimension set explicitly |
| **Offline lexical** | feature hashing, word overlap only, not semantic (default without a key) |

Minimal `.env` for Gemini:

```env
GEMINI_API_KEY=your-key
GEMINI_MODEL=gemini-flash-latest          # alias for the current stable Flash model; pin a version if needed
```

Minimal `.env` for ShopAIKey answers (search stays offline or Gemini):

```env
DOCUMIND_LLM_PROVIDER=shopaikey
SHOPAIKEY_API_KEY=sk-...
SHOPAIKEY_PROTOCOL=openai                 # openai | anthropic | gemini
SHOPAIKEY_MODEL=<exact model id from your key's group>
```

You can also switch provider, protocol and model in **Settings → Answer generation**, paste a key for the
session only, load the gateway's model catalog and run **Test connection**. The status changes to
**Connected** only after a real request succeeds, and auth, balance, model-access and rate-limit failures
are reported as such. ShopAIKey is a third-party gateway: with it enabled, your question and the retrieved
excerpts leave this machine.

All settings, including output-token, timeout and retry limits, are documented in
[`.env.example`](.env.example). Model names are configuration, not code. Without any key the app still
uploads, indexes, searches and evaluates; chat replies are labelled as retrieval only.

> **Changing the embedding provider, model or dimension** creates a new vector collection. Existing documents
> are marked *Needs re-index* and excluded from search until you click **Re-index**; vectors from different
> models are never compared.

## Using it

1. **Documents**: upload PDF or Word files, or click **Load sample documents**. Each row shows type,
   pages or paragraphs, chunks and status. The **⋯** menu opens **Inspect chunks** (exactly what was indexed),
   **Re-index** (after a model change) and **Delete** (removes the document, its chunk snapshot and all of its
   vectors, after confirmation).
2. **Chat**: pick documents with the chips above the composer and ask in English or Vietnamese. Expand
   **Sources** to read each passage with its page (PDF) or section and paragraph (DOCX).
3. **Evaluation**: run the benchmark on the sample set and inspect failure cases.
4. **Settings**: model status, **Test connection** (verifies the key and models without spending tokens),
   session-only API key, retrieval parameters and storage.

Example answer format (with Gemini configured):

> Full-time employees receive 18 working days of paid annual leave per year [S1].
>
> **[S1] Employee Handbook.pdf — Page 3** → *"Full-time employees are entitled to 18 working days of paid annual leave per calendar year…"*

## Demo dataset

`demo/generate_demo.py` reproducibly generates three fictional documents for the fictional company
*Veridian Analytics Ltd.*:

| File | Content |
|---|---|
| `Employee Handbook.pdf` (7 pages) | working hours, annual & sick leave, remote work, expenses, internal procedures |
| `Product Technical Specification.pdf` (7 pages) | architecture, performance, compatibility, authentication, retention, security |
| `Company Information Security Policy.docx` | classification table, passwords, access control, data handling, incident response, retention table |

The documents deliberately contain **conflicts** (audit-log retention 12 vs 24 months; incident reporting within
24 hours vs 1 hour; access reviews every 6 months vs quarterly) and a **synthetic prompt-injection string** to
exercise grounding. [`demo/questions.json`](demo/questions.json) holds 20 labelled questions.

## Evaluation

```powershell
.\.venv\Scripts\python.exe demo/run_evaluation.py                # offline: lexical embeddings, retrieval metrics
.\.venv\Scripts\python.exe demo/run_evaluation.py --mode gemini  # Gemini embeddings + answers (needs key)
```

Measured result of the **offline lexical baseline** (top-k = 5, 20 questions, 16 answerable;
[`docs/evaluation_results/offline.json`](docs/evaluation_results/offline.json)):

| Recall@5 | Hit@5 | MRR | Source identification | Out-of-scope leaks | Recall@5 EN | Recall@5 VI |
|---|---|---|---|---|---|---|
| 0.69 | 0.81 | 0.72 | 0.81 | 0 | 0.79 | 0.38 |

This baseline exists to test the pipeline without an API key. It matches words, not meaning, so it largely
fails Vietnamese questions over English documents. **The Gemini-embedding and answer-level evaluation has not
been run yet** (it needs an API key); see [docs/EVALUATION.md](docs/EVALUATION.md) for metric definitions,
failure analysis and how to run it.

## Tests

```powershell
scripts\test.ps1            # ruff lint + format check + offline pytest
scripts\test.ps1 -Live      # also the live Gemini tests (needs GEMINI_API_KEY)
```

155 offline tests cover PDF/DOCX extraction, Unicode normalisation, chunk boundaries and metadata, upload
validation and path traversal, duplicates, deletion, embedding retries/rate limits (fake SDK client), ChromaDB
filtering and incompatible-model detection, citation mapping and fabricated-citation removal, prompt
construction and injection escaping, insufficient-evidence handling, missing API keys, the evaluation
metrics, the UI's citation-rendering helpers, and the provider layer: all three ShopAIKey adapters against
fake SDK clients (request shape, usage parsing, parameter fallback, error states, bounded retries), gateway
URL safety, key separation and redaction, model discovery, ShopAIKey embeddings and provider switching.
Live tests (`tests/test_live_gemini.py`, `tests/test_live_shopaikey.py`) are marked `live`, skipped without
keys and excluded by default. CI runs lint and tests on Ubuntu and Windows.

## Security

Uploaded content is treated as **untrusted data**: passages are fenced in `<source>` tags with tag-like text
escaped, and the system prompt tells the model to ignore instructions inside them. Files are validated by
extension *and* signature, size-limited, processed in memory (never written to disk), and filenames are
sanitised against path traversal and Windows device names. Keys live in `.env` / Streamlit secrets, are
excluded from `repr` and redacted from logs. Retrieval is filtered by document ID in ChromaDB *and* re-checked in
Python. The ShopAIKey key is only ever sent to the configured HTTPS gateway (official vendor hosts are
refused). Details and limits: [docs/SECURITY.md](docs/SECURITY.md).

## Project skills

Two project-local [Claude Code skills](https://docs.claude.com/en/docs/claude-code/skills) are committed so
future work on the UI follows the same rules:

| Skill | Purpose |
|---|---|
| [`minimalist-chat-ui`](.claude/skills/minimalist-chat-ui/SKILL.md) | Design philosophy, tokens, typography, layout, component rules, anti-pattern checklist and Streamlit implementation rules |
| [`streamlit-visual-qa`](.claude/skills/streamlit-visual-qa/SKILL.md) | Viewports, states to capture, accessibility, rerun behaviour and regression steps to run before committing UI changes |

The redesign also followed the guidance of Anthropic's official `frontend-design` skill
(`anthropics/claude-plugins-official`): plan tokens first, review the plan for generic choices, then
self-critique against real screenshots.

## Deployment

DocuMind is designed as a **single-user local application**. It can run on Streamlit Community Cloud with
`DOCUMIND_SESSION_ISOLATION=true` (each browser session gets a private, temporary knowledge base) — read
[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) first: a public app with your own API key lets anyone spend your quota.
No public deployment is maintained for this repository.

## Known limitations

- No OCR — scanned PDFs are detected and flagged, not read.
- No user accounts; session isolation is the only multi-user safeguard.
- Similarity scores are cosine similarities, not calibrated confidence.
- Grounding and citation validation reduce, but cannot eliminate, model errors. A valid citation proves the
  source exists, not that the sentence faithfully paraphrases it.
- Heading detection in DOCX relies on Word heading styles.
- Gemini and ShopAIKey answer quality has not been benchmarked in this repository yet, and the ShopAIKey
  adapters have only been tested against fake SDK clients. No live ShopAIKey request has been made yet.

## Roadmap

OCR for scanned PDFs · hybrid (BM25 + vector) search and reranking · streaming answers · LLM-judged faithfulness
evaluation · FastAPI endpoints · authentication for multi-user hosting · in-app PDF viewer with highlights.

## License

[MIT](LICENSE)
