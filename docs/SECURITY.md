# Security

DocuMind processes documents that may be confidential and passes their text to an external LLM API. This page
lists the controls that are implemented, and what they do **not** protect against.

## Implemented controls

| Threat | Control | Where |
|---|---|---|
| Oversized uploads / memory exhaustion | Size limit (`MAX_UPLOAD_MB`, default 20) checked before parsing; Streamlit `maxUploadSize` | `document_service.validate_upload`, `.streamlit/config.toml` |
| Disguised file types | Extension allow-list (`.pdf`, `.docx`) **and** content signature (`%PDF-`; ZIP containing `word/document.xml`) | `file_utils.detect_file_type` |
| Zip bombs | DOCX rejected if its total uncompressed size exceeds 250 MB | `file_utils.detect_file_type` |
| Path traversal / unsafe names | Directory parts stripped (both `/` and `\`), reserved characters replaced, Windows device names (`CON`, `NUL`, …) prefixed, length capped | `file_utils.sanitize_filename` |
| Temporary-file leakage | Uploads are parsed from memory; original files are never written to disk | `pdf_reader`, `docx_reader` |
| Prompt injection via documents | Passages fenced in `<source>` blocks; `<source`, `</sources`, `<system`, `<question` inside document text are escaped; system prompt states document text is untrusted and must not be obeyed | `llm/prompts.py` |
| Fabricated citations | Citation IDs are issued by the app; unknown IDs are removed from the displayed answer and reported | `citations/citation_manager.py` |
| Cross-document leakage | Chroma `where` filter on selected IDs **plus** a Python re-check of every returned chunk; empty selection searches nothing | `vector_store.query`, `retriever.retrieve` |
| Stale data after deletion | Delete removes vectors from **every** DocuMind collection, the chunk snapshot and the registry entry | `ingestion_service.delete`, `vector_store.delete_document` |
| Incompatible vectors | Collections are keyed by embedding model + dimension and verified on open | `vector_store.VectorStore` |
| Secret exposure | `.env` / `secrets.toml` git-ignored; key excluded from `Settings.__repr__`; log filter redacts `AIza…` keys and `api_key=` pairs; API errors are mapped to fixed messages | `config.py`, `logging_utils.py`, `llm/errors.py` |
| Raw exception disclosure | Extraction, embedding and generation errors are converted to user-facing messages without stack traces or request details | services, `llm/errors.py` |
| Cross-user access (hosted) | `DOCUMIND_SESSION_ISOLATION=true` gives each browser session its own temporary knowledge base | `app.py` |

Tests covering these controls live in `tests/test_document_service.py`, `tests/test_rag.py`,
`tests/test_retrieval.py` and `tests/test_citations.py`.

## Limitations — read before using real data

- **Data leaves your machine.** With Gemini enabled, retrieved passages and questions are sent to Google's API
  and are subject to Google's terms for your account tier. Do not upload data you are not allowed to send to a
  third-party processor.
- **Prompt-injection defences are mitigations, not guarantees.** A sufficiently adversarial document can still
  influence an LLM. Citations let a reader check the evidence, but the reader must actually check it.
- **No authentication or authorisation.** Anyone who can reach the Streamlit port can use the app and see the
  knowledge base. The Windows launcher binds to `127.0.0.1`, so the app is reachable only from your machine;
  a plain `streamlit run app.py` listens on all interfaces unless you add `--server.address 127.0.0.1`.
- **Local storage is not encrypted.** `data/chroma` contains document text in plain form. Use disk
  encryption and delete documents you no longer need.
- **Session isolation is not multi-tenancy.** It separates browser sessions on one server; it does not
  authenticate users, and anyone can open a new session. Session stores are temporary and are lost on restart.
- **Malicious PDFs.** Parsing relies on PyMuPDF; keep dependencies up to date.

## Reporting

This is a portfolio project without a security team. Please open a GitHub issue without sensitive details,
or contact the repository owner privately.
