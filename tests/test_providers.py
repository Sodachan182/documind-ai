"""Multi-provider layer: config, gateway safety, ShopAIKey adapters, errors, RAG integration.

All offline: SDK clients are replaced by small fakes. Nothing here proves that
the real ShopAIKey service works; see tests/test_live_shopaikey.py for that.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from documind.app_factory import GenerationSelection, build_app, build_embedder, build_generator
from documind.config import Settings, load_settings
from documind.embeddings.base import EmbeddingError
from documind.embeddings.local_embeddings import HashingEmbedder
from documind.embeddings.shopaikey_embeddings import ShopAIKeyEmbedder
from documind.llm.anthropic_compat import AnthropicCompatibleGenerator
from documind.llm.base import prepare_history
from documind.llm.errors import GenerationError, classify_error, should_retry
from documind.llm.gemini_client import GeminiAnswerGenerator
from documind.llm.openai_compat import OpenAICompatibleGenerator
from documind.llm.shopaikey import base_url_for, gateway_root, list_models
from documind.llm.structured import parse_answer_payload
from documind.models import Chunk, Citation, FileType
from documind.services.rag_service import RAGService
from documind.utils.logging_utils import redact

FAKE_KEY = "sk-test-0000000000000000000000"


class FakeHTTPError(Exception):
    def __init__(self, status: int, message: str = "") -> None:
        super().__init__(message or f"HTTP {status}")
        self.status_code = status
        self.message = message


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr("documind.utils.retry.time.sleep", lambda s: None)


def _sources() -> list[Citation]:
    chunk = Chunk(
        chunk_id="d:0",
        document_id="d",
        filename="Handbook.pdf",
        file_hash="h",
        file_type=FileType.PDF,
        content="Employees get 18 working days of annual leave.",
        chunk_index=0,
        page_number=3,
    )
    return [Citation(source_id="S1", chunk=chunk, similarity=0.8)]


# ----------------------------------------------------------------------------- fakes
class FakeChat:
    def __init__(self, outcomes: list) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []

    def create(self, **params):
        self.calls.append(json.loads(json.dumps(params)))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def openai_response(content: str, model: str = "gw-model") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=120, completion_tokens=30, total_tokens=150),
        model=model,
    )


def openai_gen(outcomes: list, **kw) -> tuple[OpenAICompatibleGenerator, FakeChat]:
    chat = FakeChat(outcomes)
    client = SimpleNamespace(chat=SimpleNamespace(completions=chat))
    gen = OpenAICompatibleGenerator(
        "", "gw-model", base_url="https://api.shopaikey.com/v1", client=client, max_retries=kw.pop("max_retries", 2)
    )
    return gen, chat


# ----------------------------------------------------------------------------- config
def test_llm_provider_auto_resolution() -> None:
    assert Settings().resolved_llm_provider() == "offline"
    assert Settings(gemini_api_key="g").resolved_llm_provider() == "gemini"
    assert Settings(shopaikey_api_key=FAKE_KEY).resolved_llm_provider() == "shopaikey"
    assert Settings(gemini_api_key="g", llm_provider="shopaikey").resolved_llm_provider() == "shopaikey"


def test_env_names_and_legacy_aliases(monkeypatch) -> None:
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "DOCUMIND_EMBEDDING_PROVIDER"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DOCUMIND_LLM_PROVIDER", "ShopAIKey")
    monkeypatch.setenv("SHOPAIKEY_API_KEY", FAKE_KEY)
    monkeypatch.setenv("SHOPAIKEY_PROTOCOL", "anthropic")
    monkeypatch.setenv("SHOPAIKEY_MODEL", "claude-x")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "offline")  # legacy name + friendly alias
    monkeypatch.setenv("GEMINI_TIMEOUT_S", "33")  # legacy alias of DOCUMIND_REQUEST_TIMEOUT_S
    s = load_settings(use_dotenv=False)
    assert (s.llm_provider, s.shopaikey_protocol, s.shopaikey_model) == ("shopaikey", "anthropic", "claude-x")
    assert s.embedding_provider == "local" and s.request_timeout_s == 33
    assert FAKE_KEY not in repr(s)


def test_new_env_name_wins_over_legacy(monkeypatch) -> None:
    monkeypatch.setenv("DOCUMIND_EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    assert load_settings(use_dotenv=False).embedding_provider == "gemini"


# ----------------------------------------------------------------------------- gateway safety
@pytest.mark.parametrize(
    ("url", "root"),
    [
        ("https://api.shopaikey.com", "https://api.shopaikey.com"),
        ("https://api.shopaikey.com/v1", "https://api.shopaikey.com"),
        ("https://api.shopaikey.com/v1/", "https://api.shopaikey.com"),
        (None, "https://api.shopaikey.com"),
    ],
)
def test_gateway_root_normalises(url, root) -> None:
    assert gateway_root(url) == root


@pytest.mark.parametrize(
    "url",
    [
        "http://api.shopaikey.com",
        "https://api.openai.com/v1",
        "https://api.anthropic.com",
        "https://generativelanguage.googleapis.com",
    ],
)
def test_gateway_root_refuses_insecure_or_official_hosts(url) -> None:
    with pytest.raises(GenerationError):
        gateway_root(url)


def test_base_url_per_protocol() -> None:
    root = "https://api.shopaikey.com"
    assert base_url_for("openai", root) == "https://api.shopaikey.com/v1"
    assert base_url_for("anthropic", root) == root
    assert base_url_for("gemini", root) == root


def test_real_sdk_clients_point_at_the_gateway(monkeypatch) -> None:
    """Constructing real SDK clients makes no network call; inspect their base URLs."""
    s = Settings()
    common = {"max_output_tokens": 256, "timeout_s": 5, "max_retries": 0}
    from documind.llm.shopaikey import build_generator as sak

    oa = sak("openai", api_key=FAKE_KEY, model="m", root="https://api.shopaikey.com/v1", **common)
    assert str(oa._client.base_url).rstrip("/") == "https://api.shopaikey.com/v1"
    an = sak("anthropic", api_key=FAKE_KEY, model="m", root=s.shopaikey_base_url, **common)
    assert str(an._client.base_url).rstrip("/") == "https://api.shopaikey.com"

    captured = {}

    class FakeGenaiClient:
        def __init__(self, *, api_key, http_options):
            captured["api_key"] = api_key
            captured["base_url"] = http_options.base_url

    monkeypatch.setattr("google.genai.Client", FakeGenaiClient)
    ge = sak("gemini", api_key=FAKE_KEY, model="gemini-x", root=s.shopaikey_base_url, **common)
    assert captured == {"api_key": FAKE_KEY, "base_url": "https://api.shopaikey.com"}
    assert ge.provider == "shopaikey-gemini"


def test_anthropic_adapter_rejects_v1_base_url() -> None:
    with pytest.raises(GenerationError, match="must not end with /v1"):
        AnthropicCompatibleGenerator(FAKE_KEY, "m", base_url="https://api.shopaikey.com/v1")


def test_selection_keeps_keys_separate() -> None:
    s = Settings(
        gemini_api_key="AIzaGOOGLEKEY", shopaikey_api_key=FAKE_KEY, shopaikey_model="m", llm_provider="shopaikey"
    )
    sel = GenerationSelection.from_settings(s)
    gen, notice = build_generator(s, sel)
    assert notice is None and gen.provider == "shopaikey-openai"
    assert gen._client.api_key == FAKE_KEY  # the Google key is never handed to the gateway client
    gem, _ = build_generator(s, GenerationSelection(provider="gemini", gemini_key=None, gemini_model="x"))
    assert gem is None  # Gemini direct never falls back to the ShopAIKey key


def test_incomplete_shopaikey_selection_is_reported() -> None:
    s = Settings()
    gen, notice = build_generator(s, GenerationSelection(provider="shopaikey", shopaikey_key=FAKE_KEY))
    assert gen is None and "no model" in notice
    gen, notice = build_generator(s, GenerationSelection(provider="shopaikey", shopaikey_model="m"))
    assert gen is None and "API key" in notice


def test_provider_switching_reuses_the_same_knowledge_base(settings, sample_pdf) -> None:
    from documind.app_factory import with_generator
    from tests.conftest import FakeGenerator

    app = build_app(settings, api_key="", embedder=HashingEmbedder())
    doc_id = app.ingestion.ingest("p.pdf", sample_pdf).record.document_id
    a = with_generator(app, FakeGenerator("Twelve days [S1]."))
    b = with_generator(app, None)
    assert a.store is b.store is app.store
    assert a.rag.ask("annual leave?", [doc_id]).mode == "generated"
    assert b.rag.ask("annual leave?", [doc_id]).mode == "retrieval_only"


# ----------------------------------------------------------------------------- OpenAI-compatible
def test_openai_adapter_builds_grounded_messages_and_parses_usage() -> None:
    payload = json.dumps({"answer": "18 days [S1].", "insufficient_evidence": False})
    gen, chat = openai_gen([openai_response(payload)])
    history = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello [S2]"}]
    out = gen.generate("How many days?", _sources(), "en", history=history)
    call = chat.calls[0]
    roles = [m["role"] for m in call["messages"]]
    assert roles == ["system", "user", "assistant", "user"]
    assert "untrusted" in call["messages"][0]["content"]
    assert '<source id="S1"' in call["messages"][-1]["content"]
    assert call["messages"][2]["content"] == "Hello"  # old citation markers stripped
    assert call["max_tokens"] == 1024 and call["model"] == "gw-model"
    assert out.answer == "18 days [S1]." and out.usage.total_tokens == 150


def test_openai_adapter_accepts_fenced_json_and_plain_text() -> None:
    fenced = '```json\n{"answer": "Yes [S1].", "insufficient_evidence": false}\n```'
    gen, _ = openai_gen([openai_response(fenced), openai_response("Plain answer [S1].")])
    assert gen.generate("q", _sources(), "en").answer == "Yes [S1]."
    assert gen.generate("q", _sources(), "en").answer == "Plain answer [S1]."


def test_openai_adapter_switches_token_parameter_once() -> None:
    err = FakeHTTPError(400, "Unsupported parameter: 'max_tokens'. Use 'max_completion_tokens' instead.")
    ok = openai_response(json.dumps({"answer": "x [S1]", "insufficient_evidence": False}))
    gen, chat = openai_gen([err, ok])
    gen.generate("q", _sources(), "en")
    assert "max_tokens" in chat.calls[0] and "max_completion_tokens" in chat.calls[1]


def test_openai_adapter_drops_unsupported_temperature() -> None:
    err = FakeHTTPError(400, "Unsupported value: 'temperature' does not support 0.2")
    ok = openai_response(json.dumps({"answer": "x [S1]", "insufficient_evidence": False}))
    gen, chat = openai_gen([err, ok])
    gen.generate("q", _sources(), "en")
    assert "temperature" in chat.calls[0] and "temperature" not in chat.calls[1]


@pytest.mark.parametrize(
    ("error", "state", "calls"),
    [
        (FakeHTTPError(401, "Invalid token"), "auth_failed", 1),
        (FakeHTTPError(403, "token quota is not enough"), "insufficient_balance", 1),
        (FakeHTTPError(503, "No available channel for model x under group default"), "model_unavailable", 1),
        (FakeHTTPError(404, "model not found"), "model_unavailable", 1),
        (FakeHTTPError(429, "The upstream is overloaded for this group"), "rate_limited", 3),
        (FakeHTTPError(502, "bad gateway"), "service_error", 3),
    ],
)
def test_openai_adapter_error_states_and_bounded_retries(error, state, calls) -> None:
    gen, chat = openai_gen([error] * 5, max_retries=2)
    with pytest.raises(GenerationError) as info:
        gen.generate("q", _sources(), "en")
    assert info.value.state == state
    assert len(chat.calls) == calls  # never more than max_retries + 1; none for non-transient errors


def test_openai_check_makes_one_tiny_request() -> None:
    gen, chat = openai_gen([openai_response("OK", model="gw-model-2026")])
    assert gen.check() == "gw-model-2026"
    assert chat.calls[0]["max_tokens"] <= 16 and len(chat.calls[0]["messages"]) == 1


def test_empty_completion_is_an_error() -> None:
    gen, _ = openai_gen([openai_response("")])
    with pytest.raises(GenerationError, match="empty"):
        gen.generate("q", _sources(), "en")


# ----------------------------------------------------------------------------- Anthropic-compatible
class FakeMessages:
    def __init__(self, outcomes: list) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []

    def create(self, **params):
        self.calls.append(params)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_anthropic_adapter_request_and_response_shape() -> None:
    text = json.dumps({"answer": "18 days [S1].", "insufficient_evidence": False})
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text[:10]), SimpleNamespace(type="text", text=text[10:])],
        usage=SimpleNamespace(input_tokens=200, output_tokens=20),
        model="claude-x",
    )
    messages = FakeMessages([response])
    gen = AnthropicCompatibleGenerator(
        "", "claude-x", base_url="https://api.shopaikey.com", client=SimpleNamespace(messages=messages)
    )
    out = gen.generate("q", _sources(), "vi", history=[{"role": "assistant", "content": "orphan"}])
    call = messages.calls[0]
    assert "Vietnamese" in call["system"] and call["max_tokens"] == 1024
    assert [m["role"] for m in call["messages"]] == ["user"]  # leading assistant turn dropped
    assert "temperature" not in call
    assert out.answer == "18 days [S1]." and out.usage.total_tokens == 220


# ----------------------------------------------------------------------------- Gemini-compatible
def test_gemini_compatible_check_uses_real_generation_not_models_get() -> None:
    class Models:
        def __init__(self) -> None:
            self.generated = 0

        def get(self, **kw):
            raise AssertionError("models.get is not proof of access through the gateway")

        def generate_content(self, **kw):
            self.generated += 1
            assert kw["config"].max_output_tokens <= 16
            return SimpleNamespace(model_version="gemini-x", text="OK")

    models = Models()
    gen = GeminiAnswerGenerator("", "gemini-x", provider="shopaikey-gemini", client=SimpleNamespace(models=models))
    assert gen.check() == "gemini-x" and models.generated == 1


def test_gemini_history_uses_model_role() -> None:
    seen = {}

    class Models:
        def generate_content(self, **kw):
            seen["contents"] = kw["contents"]
            return SimpleNamespace(parsed=None, text=json.dumps({"answer": "a [S1]", "insufficient_evidence": False}))

    gen = GeminiAnswerGenerator("", "g", client=SimpleNamespace(models=Models()))
    gen.generate(
        "q", _sources(), "en", history=[{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]
    )
    assert [c["role"] for c in seen["contents"]] == ["user", "model", "user"]


# ----------------------------------------------------------------------------- model discovery
def test_list_models_openai_and_anthropic_filter() -> None:
    page = SimpleNamespace(data=[SimpleNamespace(id=i) for i in ("gpt-5-mini", "claude-sonnet-4-5", "gpt-5-mini")])
    client = SimpleNamespace(models=SimpleNamespace(list=lambda: page))
    assert list_models("openai", api_key="", root="https://api.shopaikey.com", client=client) == [
        "claude-sonnet-4-5",
        "gpt-5-mini",
    ]
    assert list_models("anthropic", api_key="", root="https://api.shopaikey.com", client=client) == [
        "claude-sonnet-4-5"
    ]


def test_list_models_gemini_strips_prefix() -> None:
    items = [SimpleNamespace(name="models/gemini-2.5-flash"), SimpleNamespace(name="models/gemini-2.5-pro")]
    client = SimpleNamespace(models=SimpleNamespace(list=lambda: iter(items)))
    assert list_models("gemini", api_key="", root="https://api.shopaikey.com", client=client) == [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
    ]


def test_list_models_errors_are_classified() -> None:
    def boom():
        raise FakeHTTPError(401, "Invalid token")

    client = SimpleNamespace(models=SimpleNamespace(list=boom))
    with pytest.raises(GenerationError) as info:
        list_models("openai", api_key="", root="https://api.shopaikey.com", client=client)
    assert info.value.state == "auth_failed"


# ----------------------------------------------------------------------------- embeddings
def _emb_client(dims: int, n: int | None = None):
    def create(model, input):
        count = n if n is not None else len(input)
        data = [SimpleNamespace(index=i, embedding=[3.0, 4.0] + [0.0] * (dims - 2)) for i in reversed(range(count))]
        return SimpleNamespace(data=data)

    return SimpleNamespace(embeddings=SimpleNamespace(create=create))


def test_shopaikey_embedder_normalises_and_keys_vector_space() -> None:
    emb = ShopAIKeyEmbedder("", "text-embedding-x", 8, base_url="https://api.shopaikey.com/v1", client=_emb_client(8))
    vec = emb.embed_query("hello")
    assert vec[:2] == pytest.approx([0.6, 0.8]) and len(vec) == 8
    assert emb.key == "shopaikey:text-embedding-x:8"


def test_shopaikey_embedder_dimension_mismatch_fails_loudly() -> None:
    emb = ShopAIKeyEmbedder("", "m", 1536, base_url="https://x.invalid/v1", client=_emb_client(8))
    with pytest.raises(EmbeddingError, match="returned 8-dim"):
        emb.embed_documents(["a"])


def test_build_embedder_requires_explicit_shopaikey_config() -> None:
    emb, notices = build_embedder(Settings(embedding_provider="shopaikey", shopaikey_api_key=FAKE_KEY), None)
    assert isinstance(emb, HashingEmbedder) and "DOCUMIND_EMBEDDING_MODEL" in notices[0]
    s = Settings(
        embedding_provider="shopaikey",
        shopaikey_api_key=FAKE_KEY,
        embedding_model="text-embedding-x",
        embedding_dimension=1536,
    )
    emb, notices = build_embedder(s, None)
    assert isinstance(emb, ShopAIKeyEmbedder) and emb.key == "shopaikey:text-embedding-x:1536" and not notices
    # Choosing ShopAIKey for answers never changes the vector space.
    assert build_embedder(Settings(llm_provider="shopaikey", shopaikey_api_key=FAKE_KEY), None)[0].provider == "local"


# ----------------------------------------------------------------------------- errors, retry, redaction
def test_should_retry_policy() -> None:
    assert should_retry(FakeHTTPError(429)) and should_retry(FakeHTTPError(500))
    assert not should_retry(FakeHTTPError(503, "No available channel for model m under group g"))
    assert not should_retry(FakeHTTPError(403, "This token quota is exhausted"))
    assert not should_retry(FakeHTTPError(400, "bad"))


def test_error_messages_never_contain_keys() -> None:
    err = FakeHTTPError(400, f"bad request with Authorization: Bearer {FAKE_KEY} and ?key=AIzaSyA1234567890abcdefghijk")
    state, message = classify_error(err, "generation", "ShopAIKey")
    assert state == "bad_request" and FAKE_KEY not in message and "AIza" not in message


@pytest.mark.parametrize(
    "text",
    [
        f"token {FAKE_KEY} leaked",
        "Authorization: Bearer abcdefghijklmnop",
        "GET /v1beta/models?key=secretvalue123&alt=sse",
        'x-api-key: "sk-abcdefghijklmnopqrstu"',
    ],
)
def test_redaction(text) -> None:
    out = redact(text)
    assert "REDACTED" in out
    for secret in (FAKE_KEY, "abcdefghijklmnop", "secretvalue123", "sk-abcdefghijklmnopqrstu"):
        assert secret not in out


# ----------------------------------------------------------------------------- history / parsing
def test_prepare_history_caps_strips_and_starts_with_user() -> None:
    turns = [{"role": "assistant", "content": "x"}] + [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i} [S{i}] " + "y" * 3000} for i in range(6)
    ]
    out = prepare_history(turns)
    assert out[0]["role"] == "user" and len(out) <= 4
    assert all("[S" not in t["content"] and len(t["content"]) <= 1500 for t in out)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"answer": "A", "insufficient_evidence": true}', ("A", True)),
        ('Sure! {"answer": "B", "insufficient_evidence": false} Thanks', ("B", False)),
        ("not json at all", ("not json at all", False)),
        ('{"other": 1}', ('{"other": 1}', False)),
    ],
)
def test_parse_answer_payload(text, expected) -> None:
    assert parse_answer_payload(text) == expected


# ----------------------------------------------------------------------------- RAG integration
def test_shopaikey_adapter_through_the_rag_pipeline_validates_citations(app, sample_pdf) -> None:
    doc_id = app.ingestion.ingest("Handbook.pdf", sample_pdf).record.document_id
    payload = json.dumps({"answer": "Twelve days of leave [S1]. Invented [S9].", "insufficient_evidence": False})
    gen, chat = openai_gen([openai_response(payload)])
    answer = RAGService(app.retriever, gen).ask(
        "annual leave days", [doc_id], history=[{"role": "user", "content": "earlier question"}]
    )
    assert answer.mode == "generated" and answer.cited_ids == ["S1"]
    assert answer.invalid_citations == ["S9"] and "S9" not in answer.answer
    assert answer.provider == "ShopAIKey (OpenAI-compatible)" and answer.usage.total_tokens == 150
    assert chat.calls[0]["messages"][1]["content"] == "earlier question"


def test_provider_error_reaches_the_answer_with_its_state(app, sample_pdf) -> None:
    doc_id = app.ingestion.ingest("Handbook.pdf", sample_pdf).record.document_id
    gen, _ = openai_gen([FakeHTTPError(403, "token quota is not enough")])
    answer = RAGService(app.retriever, gen).ask("annual leave days", [doc_id])
    assert answer.mode == "error" and answer.error_state == "insufficient_balance"
    assert answer.sources  # retrieved evidence is still shown


def test_unanswerable_with_shopaikey_adapter(app, sample_pdf) -> None:
    doc_id = app.ingestion.ingest("Handbook.pdf", sample_pdf).record.document_id
    payload = json.dumps({"answer": "", "insufficient_evidence": True})
    gen, _ = openai_gen([openai_response(payload)])
    answer = RAGService(app.retriever, gen).ask("What is the parental leave policy?", [doc_id])
    assert answer.insufficient_evidence and "could not find" in answer.answer


# ----------------------------------------------------------------------------- UI helpers
def test_history_for_model_keeps_only_generated_answers() -> None:
    from documind.ui.session import history_for_model

    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "answer": {"mode": "generated", "answer": "a1 [S1]", "insufficient_evidence": False}},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "answer": {"mode": "retrieval_only", "answer": "passages"}},
    ]
    assert history_for_model(messages) == [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1 [S1]"},
    ]


def test_openai_style_errors_with_string_code_are_classified() -> None:
    """openai/anthropic errors carry a *string* ``code``; the HTTP status is ``status_code``."""

    class SDKError(Exception):
        def __init__(self) -> None:
            super().__init__("Error code: 429")
            self.status_code = 429
            self.code = "rate_limit_exceeded"
            self.message = "The upstream is overloaded for this group"

    state, message = classify_error(SDKError(), "connection check", "ShopAIKey")
    assert state == "rate_limited" and "upstream is overloaded" in message
    assert should_retry(SDKError())
