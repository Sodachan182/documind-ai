# Deployment

DocuMind's primary target is a **local, single-user** installation (see the README quick start). This page
explains how to host it on Streamlit Community Cloud and what to consider before doing so.

> No public deployment of this repository is currently maintained.

## Before you deploy publicly

A public Streamlit app is reachable by anyone with the URL. Decide on these first:

1. **Isolation** — set `DOCUMIND_SESSION_ISOLATION=true`. Each browser session then gets its own temporary
   knowledge base, so visitors cannot see each other's documents. Without it, every visitor shares one store.
2. **API key** — if you put your `GEMINI_API_KEY` or `SHOPAIKEY_API_KEY` in secrets, every visitor spends your quota and your key's
   data terms apply to their uploads. Safer options:
   - leave the key unset — visitors paste their own key in **Settings** (kept in that session's server memory only); or
   - restrict the app to specific viewers in the Streamlit Cloud sharing settings.
3. **Data** — tell users not to upload confidential documents; this app has no authentication.

## Streamlit Community Cloud

1. Push the repository to GitHub (public or private).
2. Sign in at <https://share.streamlit.io> with GitHub and click **Create app → Deploy a public app from GitHub**.
3. Select repository `documind-ai`, branch `main`, main file `app.py`.
4. Under **Advanced settings** choose Python **3.12**. Dependencies are installed from `requirements.txt`.
5. Add secrets (TOML), for example:

   ```toml
   DOCUMIND_SESSION_ISOLATION = "true"
   # Optional - see the warning above:
   # GEMINI_API_KEY = "..."
   # GEMINI_MODEL = "gemini-flash-latest"
   # DOCUMIND_LLM_PROVIDER = "shopaikey"
   # SHOPAIKEY_API_KEY = "sk-..."
   # SHOPAIKEY_PROTOCOL = "openai"
   # SHOPAIKEY_MODEL = "<model id>"
   ```

   `app.py` copies these keys from `st.secrets` into environment variables at start-up.
6. Deploy, then open the app, click **Load sample documents** and ask a question to verify.

### Persistence

Community Cloud containers have an ephemeral filesystem. The ChromaDB directory is lost when the app restarts
or sleeps, so hosted knowledge bases are temporary by design. For durable hosting you would need an external
vector database and real user accounts — out of scope for this MVP.

## Other hosts

Any host that can run `streamlit run app.py` with Python 3.12 works. Set the same environment variables,
persist `CHROMA_PERSIST_DIR` on a volume if you need durability, and put the app behind authentication
(for example a reverse proxy with SSO) before exposing it beyond your machine.
