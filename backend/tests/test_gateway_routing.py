"""AI gateway routing tests — no database, no network, no real LLM.

1. Behavioural: send_once / collect_stream use the spec's model, max_tokens and key,
   raise on failure (call sites keep their own fallbacks) and honour timeouts.
2. Defaults: the task specs carry EXACTLY the model strings, token limits and timeouts
   that were hard-coded at the call sites before they were routed through the gateway.
3. Guard: no module except services/ai_gateway.py may construct an LlmChat or call
   .with_model(...). This is what keeps "all LLM calls go through the gateway" true.

Run alone (works without MongoDB):  pytest tests/test_gateway_routing.py --noconftest
"""
import ast
import asyncio
import os
import sys
import types
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

# config.settings reads these at import time.
for _k, _v in {"MONGO_URL": "mongodb://localhost:27017", "DB_NAME": "test",
               "EMERGENT_LLM_KEY": "test-key", "JWT_SECRET": "test-secret"}.items():
    os.environ.setdefault(_k, _v)

try:
    import emergentintegrations.llm.chat  # noqa: F401
except ImportError:  # minimal stand-in so the module under test can be imported offline
    pkg = types.ModuleType("emergentintegrations")
    llm = types.ModuleType("emergentintegrations.llm")
    chat = types.ModuleType("emergentintegrations.llm.chat")
    for name in ("LlmChat", "UserMessage", "ImageContent", "TextDelta", "StreamDone"):
        setattr(chat, name, type(name, (), {"__init__": lambda self, *a, **k: None}))
    sys.modules.update({"emergentintegrations": pkg, "emergentintegrations.llm": llm,
                        "emergentintegrations.llm.chat": chat})

from services import ai_gateway as gw  # noqa: E402


class _Msg:
    def __init__(self, text, **_):
        self.text = text


class _Delta:
    def __init__(self, content):
        self.content = content


class _Done:
    pass


class FakeChat:
    """Records how the gateway configured it."""
    instances = []
    reply = "ok"
    delay = 0.0
    fail = None
    stream_events = ()

    def __init__(self, api_key, session_id, system_message):
        self.api_key, self.session_id, self.system_message = api_key, session_id, system_message
        self.provider = self.model = self.max_tokens = self.sent = None
        FakeChat.instances.append(self)

    def with_model(self, provider, model):
        self.provider, self.model = provider, model
        return self

    def with_params(self, **kw):
        self.max_tokens = kw.get("max_tokens")
        return self

    async def send_message(self, msg):
        self.sent = msg.text
        if FakeChat.delay:
            await asyncio.sleep(FakeChat.delay)
        if FakeChat.fail:
            raise FakeChat.fail
        return FakeChat.reply

    async def stream_message(self, msg):
        self.sent = msg.text
        if FakeChat.fail:
            raise FakeChat.fail
        for ev in FakeChat.stream_events:
            yield ev


@pytest.fixture(autouse=True)
def fake_llm(monkeypatch):
    FakeChat.instances, FakeChat.reply, FakeChat.delay = [], "ok", 0.0
    FakeChat.fail, FakeChat.stream_events = None, ()
    monkeypatch.setattr(gw, "LlmChat", FakeChat)
    monkeypatch.setattr(gw, "UserMessage", _Msg)
    monkeypatch.setattr(gw, "TextDelta", _Delta)
    monkeypatch.setattr(gw, "StreamDone", _Done)


def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def test_send_once_applies_spec_model_tokens_and_key():
    out = run(gw.send_once("sid", "sys", "hello", spec=gw.FIR_EXTRACT_SPEC, caller="t"))
    chat = FakeChat.instances[0]
    assert out == "ok"
    assert (chat.provider, chat.model) == (gw.FIR_EXTRACT_SPEC.provider, gw.FIR_EXTRACT_SPEC.model)
    assert chat.max_tokens == 600
    assert chat.sent == "hello" and chat.system_message == "sys"
    assert chat.api_key == gw._api_key()


def test_send_once_without_max_tokens_does_not_set_params():
    run(gw.send_once("sid", "sys", "x", spec=gw.QUERY_PARSE_SPEC, caller="t"))
    assert FakeChat.instances[0].max_tokens is None


def test_send_once_returns_empty_string_for_none_reply():
    FakeChat.reply = None
    assert run(gw.send_once("s", "s", "x", spec=gw.FOLLOWUP_SPEC)) == ""


def test_send_once_reraises_so_call_sites_keep_their_fallbacks():
    FakeChat.fail = RuntimeError("provider down")
    with pytest.raises(RuntimeError, match="provider down"):
        run(gw.send_once("s", "s", "x", spec=gw.FIR_DRAFT_SPEC))


def test_send_once_timeout_raises_asyncio_timeout():
    FakeChat.delay = 0.3
    with pytest.raises(asyncio.TimeoutError):
        run(gw.send_once("s", "s", "x", spec=gw.QUERY_PARSE_SPEC, timeout_s=0.01))


def test_collect_stream_joins_deltas_and_stops_at_done():
    FakeChat.stream_events = (_Delta("Hel"), _Delta("lo"), _Done(), _Delta("IGNORED"))
    out = run(gw.collect_stream("s", "sys", "hi", spec=gw.VISITOR_TRANSLATE_SPEC, caller="t"))
    assert out == "Hello"


def test_collect_stream_reraises():
    FakeChat.fail = ValueError("boom")
    with pytest.raises(ValueError):
        run(gw.collect_stream("s", "s", "x", spec=gw.VISITOR_TRANSLATE_SPEC))


# ── defaults must equal what the call sites hard-coded before ─────────────────

@pytest.mark.skipif(any(os.getenv(k) for k in (
    "TASK_PROVIDER", "QUERY_PARSE_MODEL", "RETRIEVAL_KEYWORD_MODEL", "FIR_EXTRACT_MODEL",
    "FIR_DRAFT_MODEL", "FOLLOWUP_MODEL", "VISITOR_TRANSLATE_MODEL", "INTAKE_SUMMARY_MODEL")),
    reason="a task-model env override is set")
def test_task_spec_defaults_match_previous_hardcoded_values():
    assert gw.QUERY_PARSE_SPEC.provider == "anthropic"
    assert gw.QUERY_PARSE_SPEC.model == "claude-haiku-4-5" and gw.QUERY_PARSE_SPEC.timeout_s == 10.0
    assert gw.RETRIEVAL_KEYWORD_SPEC.model == "claude-haiku-4-5" and gw.RETRIEVAL_KEYWORD_SPEC.timeout_s == 12.0
    assert gw.FIR_EXTRACT_SPEC.model == "claude-haiku-4-5" and gw.FIR_EXTRACT_SPEC.max_tokens == 600
    assert gw.FIR_CORRECTION_SPEC.model == "claude-haiku-4-5" and gw.FIR_CORRECTION_SPEC.max_tokens == 400
    assert gw.FIR_DRAFT_SPEC.model == "claude-sonnet-4-6" and gw.FIR_DRAFT_SPEC.max_tokens == 2500
    assert gw.FOLLOWUP_SPEC.model == "claude-sonnet-4-5-20250929"
    assert gw.VISITOR_TRANSLATE_SPEC.model == "claude-haiku-4-5-20251001"
    assert gw.INTAKE_SUMMARY_SPEC.model == "claude-sonnet-4-5-20250929"


# ── guard: nothing but the gateway may talk to a chat model directly ──────────

def _python_files():
    for p in BACKEND.rglob("*.py"):
        rel = p.relative_to(BACKEND).as_posix()
        if rel == "services/ai_gateway.py" or rel.startswith("tests/"):
            continue
        yield p, rel


def test_no_direct_llmchat_or_with_model_outside_gateway():
    offenders = []
    for path, rel in _python_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                f = node.func
                if isinstance(f, ast.Attribute) and f.attr == "with_model":
                    offenders.append(f"{rel}:{node.lineno} .with_model(...)")
                if isinstance(f, ast.Name) and f.id == "LlmChat":
                    offenders.append(f"{rel}:{node.lineno} LlmChat(...)")
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("emergentintegrations"):
                if any(a.name == "LlmChat" for a in node.names):
                    offenders.append(f"{rel}:{node.lineno} imports LlmChat")
    assert not offenders, "Route these through services/ai_gateway.py:\n" + "\n".join(offenders)
