"""Chat, voice, bookmarks, and notification routes.

Extracted from server.py (Phase 1 Modularization).
Depends on: dependencies.py, config/settings.py
"""
import io
import logging
import os
import re
import json
import uuid
import asyncio
from datetime import datetime, timezone
from typing import List, Optional, AsyncGenerator

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form, Header, Request, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from openai import AsyncOpenAI

from emergentintegrations.llm.chat import UserMessage, TextDelta, StreamDone

from dependencies import (
    db, corpus_db, current_user, current_user_optional,
    meter_llm_use, build_system_prompt, translate_for_retrieval, logger,
)
from config.settings import (
    SERVER_CHAT_PROVIDER, SERVER_CHAT_MODEL, EMERGENT_LLM_KEY, LANGUAGES,
    PRO_FREE_SAMPLES, DRAFTS_FREE,
    PRO_PRICE_INR, PRO_PRICE_LABEL, PRO_PRICE_USD, PRO_PRICE_USD_LABEL,
)
from script_guard import check_script_mismatch
from states import state_name
from engine.pipeline import (
    run_pre_retrieval, run_post_retrieval,
    save_case_state, classify_turn,
)
from corpus import (
    retrieve as corpus_retrieve,
    retrieve_state as corpus_retrieve_state,
    state_sensitive_topic,
    is_non_indian_jurisdiction,
    is_non_legal_advice,
    public_citation,
    with_historical_notice,
    sanitize_model_output,
    REFUSAL_NO_CORPUS,
    REFUSAL_NON_INDIAN,
    REFUSAL_NOT_LEGAL,
    localize_refusal,
    top_candidate_debug,
    classify_topic,
)
from corpus_db import (
    retrieve_db as db_retrieve,
    lookup_section as db_lookup_section,
    check_query_for_orphan_warnings as db_orphan_check,
    check_bns_family_identifier as db_check_bns_family_identifier,  # P1 fix
    STATE_CODE_TO_JURISDICTION,
)
from langpolicy import needs_language_repair, repair_prompt, needs_retrieval_translation
from personal_law import (
    classify_personal_law_topic,
    detect_context as detect_personal_law_context,
    act_hint_phrase,
    disambiguation_question,
    act_disclaimer,
)
from push import register_device, unregister_device
from services.ai_gateway import build_chat as _ai_build_chat, gateway_status
from engine.answer_generator import (
    all_provisions_dead,
    build_cannot_verify_response,
    build_bns_identifier_hypothesis_response,
    augment_prompt_with_status_guard,
    augment_prompt_with_exact_dead_citation_lock,
    augment_prompt_with_numeric_fidelity_lock,
    correct_punishment_mismatch,
    strip_leaked_citations,
    augment_prompt_with_actor_conduct_lock,
    reframe_misdirected_obligations,
    recheck_dead_with_successor_release,
)
from engine.approval_evidence import check_section_correspondence_approval as _check_corr_approval
# AQF Phase 1b — jurisdiction-state unification (FEATURE_MATTER_CONTEXT_V2)
from engine.models import FEATURE_MATTER_CONTEXT_V2, resolve_jurisdiction_state
from engine.output_guard import validate_dhara_text, looks_like_refusal, CANNOT_CONFIRM_MESSAGE

# DHARA P0 Legal-Answer-Engine audit — Phase 1 hotfix flag. Default OFF =
# today's behaviour byte-for-byte (citations emitted immediately, streamed
# token-by-token, no output guard). ON = citations only after the answer
# is known to be substantive, answer streamed only after it passes the
# output guard. See backend/.env CITATION_GUARD_ENABLED.
CITATION_GUARD_ENABLED = os.getenv("CITATION_GUARD_ENABLED", "false").strip().lower() == "true"

# P0-D: reuse the single P0-B/P0-C exact-statute parser — no duplicate engine.
from retrieval_logic import extract_exact_statute_ref as _p0d_extract_exact_ref

# Canonical legacy-Act -> successor-Act-family map, used by the "sticky"
# dead-law exemption (see _has_explicit_dead_citation_with_successor) to
# recognise a follow-up turn's own successor-act hits (e.g. BNS) as
# legitimately supporting an active reference to the now-superseded Act
# (e.g. IPC) it replaced — not a new/duplicate lookup table, just the
# well-known 2023 renumbering mapping.
_LEGACY_TO_SUCCESSOR_FAMILY = {
    "indian penal code": "Bharatiya Nyaya Sanhita",
    "code of criminal procedure": "Bharatiya Nagarik Suraksha Sanhita",
    "indian evidence act": "Bharatiya Sakshya Adhiniyam",
}

router = APIRouter()

# ── DHARA C1 — conversational replies (non-legal allowlist) ──────────────────
# Default OFF. Deterministic matcher, no LLM, deny-list always wins, EN/HI
# templates only. Full contract: engine/conversational_intents.py.
# Greetings / thanks / capability replies ("good morning", "hi", "thank you")
# are a core UX feature. Keep them ON by default and fail SAFE: only an
# EXPLICIT falsy value disables them. A missing OR EMPTY env var (which the old
# `== "true"` check treated as off) must NOT silently turn greetings into "cannot
# answer" refusals in a deployed build.
FEATURE_C1_CONVERSATIONAL_REPLIES = (
    os.getenv("FEATURE_C1_CONVERSATIONAL_REPLIES", "true").strip().lower()
    not in ("false", "0", "no", "off")
)
from engine.conversational_intents import (
    classify_conversational_intent,
    get_conversational_template,
    is_denylisted,
    is_bare_distress,
)


# ── Pydantic models ───────────────────────────────────────────────────────────

# ── DHARA-P0-4 (B024): immediate-danger guard ─────────────────────────────────────────
# A user in danger right now was told to "go to the Magistrate's court right
# now" with no mention of 112, and was promised police/shelter outcomes the
# cited source does not support. When the message signals present danger we
# (1) always lead with 112 via a fixed banner, (2) instruct the model to give
# safety steps first and make no outcome promises, and (3) drop domestic-
# violence citations unless the user actually described a domestic relation.
_DANGER_RE = re.compile(
    r"(right now|at this moment|happening now|not safe|unsafe|in danger|"
    r"threaten(ing|s|ed)? (me|to kill)|going to (kill|hurt|attack) me|attacking me|beating me|hitting me|"
    r"kill me|help me now|save me|अभी|इसी समय|खतरे में|सुरक्षित नहीं|मार रहा|मार रही|मार देगा|जान से मार|बचाओ|"
    r"आता|धोक्यात|सुरक्षित नाही|मारत आहे|वाचवा|இப்போது|ஆபத்தில்|பாதுகாப்பாக இல்லை|காப்பாற்றுங்கள்)",
    re.I,
)
_THREAT_RE = re.compile(
    r"(\bthreat|\battack|\bbeat(ing|s|en)?\b|\bhit(ting|s)?\b|\bhurt|\bkill|\bweapon|\bknife|unsafe|not safe|\bdanger|\babus|\bviolen|\bchasing|following me|"
    r"धमकी|मार|हमला|खतरा|खतरे|धोका|धोक्यात|அச்சுறுத்த|ஆபத்து|தாக்க)",
    re.I,
)
_DOMESTIC_FACT_RE = re.compile(
    r"(husband|wife|in-?laws?|partner|spouse|dowry|domestic|marri|पति|पत्नी|ससुराल|दहेज|नवरा|बायको|கணவன்|மனைவி|வரதட்சணை)",
    re.I,
)
_DV_ACT_RE = re.compile(r"(PWDVA|Domestic Violence)", re.I)

# Acute-danger phrases that indicate present danger ON THEIR OWN (no separate
# time/state phrase needed): an armed person, being followed/chased right now,
# or a direct "will kill / save me" in Hindi (Devanagari + romanised).
# Kept narrow on purpose: legal hypotheticals ("punishment for carrying a
# knife") and "my case is being followed up" must NOT match.
_ACUTE_DANGER_RE = re.compile(
    r"(\b(he|she|they|someone|somebody|a man|a woman|a guy|my husband|husband|boyfriend)\s+"
    r"(has|have|had|is holding|is carrying|pulled out|took out|is waving)\s+(a |an )?(knife|gun|pistol|weapon|acid|blade)\b"
    r"|\b(i am|i'm|im|i am being|i'm being)\s+(being\s+)?(followed|stalked|chased)\b(?!\s+up)"
    r"|\b(someone|somebody|a man|he|they|a car|a bike)\s+(is|are|has been|keeps)\s+(following|chasing|stalking)\s+me\b"
    r"|\bmaar\s*(dega|degi|denge|dalega|dalegi|daalega|daalegi|daalenge)\b|\bjaan\s+se\s+maar"
    r"|\bmujhe\s+bachao\b|मुझे\s*बचाओ|मार\s*डालेगा|मार\s*डालेगी"
    r"|\bmala\s+vachva\b|मला\s*वाचवा"
    # B-05 proposal: present-progressive break-in / intruder (English).
    r"|\b(someone|somebody|a man|men|they|he)\s+(is|are)\s+(breaking|forcing|trying to break|trying to get)\s+(in|into)\b"
    r"|\b(just|now)\s+broke\s+into\s+my\s+(house|home|room|flat)\b"
    # B-05 proposal: romanised Hindi / Marathi "is hitting / threatening me" (native-speaker review needed).
    r"|\b(maar|peet|maarr)\s*(raha|rahi|rahe)\s*(hai|hain)\b"
    r"|\bmujhe\s+(maar|peet)\s*(raha|rahi|rahe)\b"
    r"|\bdhamki\s+det\s+(ahe|aahe|aahet)\b|\bmala\s+(maarat|marat)\s+(ahe|aahe)\b)",
    re.I,
)

# B-05 proposal: a general question about the law ("what is the punishment for
# beating someone right now") must not trigger the two-condition rule unless
# the writer puts themselves in it. Safety bias: any first-person marker keeps
# the old behaviour.
_LEGAL_QUESTION_RE = re.compile(
    r"^\s*(what\s+is\s+the\s+(punishment|penalty|law|section)|what\s+does\s+(the\s+)?(law|section)|"
    r"punishment\s+for|penalty\s+for|is\s+it\s+(a\s+)?(crime|offen[cs]e|illegal)|"
    r"what\s+happens\s+(if|when)\s+someone)",
    re.I,
)
_FIRST_PERSON_RE = re.compile(r"(\bme\b|\bmy\b|\bmine\b|\bi\b|\bi'm\b|\bim\b|\bus\b|\bour\b|mujhe|mujhko|mera|meri|मुझे|मेरा|मेरी|मला|माझ)", re.I)


def _is_immediate_danger(text: str) -> bool:
    t = text or ""
    if _ACUTE_DANGER_RE.search(t):
        return True
    if _LEGAL_QUESTION_RE.search(t) and not _FIRST_PERSON_RE.search(t):
        return False
    return bool(_DANGER_RE.search(t) and _THREAT_RE.search(t))


_EMERGENCY_PROMPT = """

IMMEDIATE-DANGER MODE (mandatory, overrides other style rules):
The user says they are in danger right now. Your reply must:
- Start with what keeps them safe now: get to a safe place or near other people if they can, and call 112 (all-India emergency). Do NOT tell them to go to a court, magistrate, or any office first.
- Never promise outcomes. Do not say the police "must" or "will" do anything, that a shelter or officer "will" help, or that any order "will" be granted.
- Do not assume who the person is, their gender, or their relationship to the threatening person. Do not assume domestic violence unless they said so.
- Only after safety, briefly mention that legal options (such as a complaint or a protection order) may be possible later depending on the facts, and that the 181 Women Helpline or 15100 (free legal aid) can guide them.
- Keep it short and calm.
"""

_EMERGENCY_BANNER = {
    "en": "🚨 If you are in danger right now, call 112 (emergency) immediately. Women Helpline: 181.",
    "hi": "🚨 अगर आप अभी खतरे में हैं, तो तुरंत 112 (आपातकालीन) पर कॉल करें। महिला हेल्पलाइन: 181।",
    "mr": "🚨 तुम्ही आत्ता धोक्यात असाल, तर लगेच 112 (आपत्कालीन) वर कॉल करा. महिला हेल्पलाइन: 181.",
    "ta": "🚨 நீங்கள் இப்போது ஆபத்தில் இருந்தால், உடனே 112 (அவசரம்) ஐ அழைக்கவும். பெண்கள் உதவி எண்: 181.",
}


# ── B-04b / B-04c: safety before access gates ────────────────────────────────
# An expired guest session (401) and an exhausted Pro sample quota (402) are
# ACCESS gates, not safety gates. A person in immediate danger — or sending a
# bare distress call — must still receive the deterministic 112 banner / SOS
# reply instead of a hard error, exactly like the quota-exhausted guest path
# (B-04a). This helper returns that deterministic SSE stream, or None when the
# message is an ordinary one (caller then raises its 401/402 unchanged).
# No LLM call, no guest-question increment, no Pro-sample consumption.
def _gate_safety_stream(message: str, language: str, emergency: bool, session_frame: dict):
    _sos_text = None
    _flag = None
    if emergency:
        _sos_text = _EMERGENCY_BANNER.get(language, _EMERGENCY_BANNER["en"])
        _flag = "emergency"
    elif FEATURE_C1_CONVERSATIONAL_REPLIES and is_bare_distress(message):
        _sos_text = (
            get_conversational_template("distress_sos", language)
            or _EMERGENCY_BANNER.get(language, _EMERGENCY_BANNER["en"])
        )
        _flag = "c1_sos"
    if not _sos_text:
        return None

    def _sse(obj: dict) -> bytes:
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8")

    async def _gen():
        yield _sse(session_frame)
        yield _sse({"type": "delta", "content": _sos_text, _flag: True})
        yield _sse({"type": "done"})

    return StreamingResponse(
        _gen(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── IT Act §66A struck-down guard (Layer S, section-level) ────────────────────
# Section 66A of the IT Act, 2000 was struck down in its entirety as
# unconstitutional in Shreya Singhal v. Union of India (2015) 5 SCC 1. The
# act-level status registry (data/law_status_registry.json, act_id "IT_2000",
# review_status SOURCED, source_url present) records this fact, but §66A has no
# section-level record — so a direct / indirect / historical query about it
# otherwise falls through to a generic "could not find active law" refusal that
# never tells the user it was struck down. This deterministic guard answers such
# queries authoritatively: no LLM call, no fabricated replacement section or
# punishment. The wording restates the sourced Supreme Court holding only and,
# like all legal content in DHARA, remains pending qualified advocate approval.
# The regex uses \b before "66" so "166A" (IPC 166A — a different provision,
# handled by the BNS-supersession path) is NOT matched.
_IT66A_RE = re.compile(r"\b66\s*-?\s*a\b", re.I)


def _is_it_66a_query(text: str) -> bool:
    return bool(_IT66A_RE.search(text or ""))


_IT66A_NOTICE = {
    "en": (
        "Section 66A of the Information Technology Act, 2000 was struck down in its "
        "entirety as unconstitutional by the Supreme Court in Shreya Singhal v. Union of "
        "India (2015) 5 SCC 1.\n\n"
        "This means Section 66A is no longer valid law. You cannot be arrested, charged or "
        "prosecuted under Section 66A for any online post, message or comment — the police "
        "cannot invoke it, and any FIR or case registered under it is liable to be quashed.\n\n"
        "Depending on the exact facts, other provisions of the IT Act or the Bharatiya Nyaya "
        "Sanhita, 2023 may apply to specific online conduct — speak to a lawyer about your "
        "situation."
    ),
}


class ChatIn(BaseModel):
    message: str
    session_id: Optional[str] = None
    language: str = "en"
    language_name: str = "English"
    language_native: Optional[str] = None
    mode: str = "basic"  # "basic" | "pro"

class ClientErrorLogIn(BaseModel):
    context: str
    error_name: Optional[str] = None
    error_message: Optional[str] = None
    platform: Optional[str] = None
    retried: Optional[bool] = None

class RegisterPushBody(BaseModel):
    user_id: str
    platform: str
    device_token: str

class TTSIn(BaseModel):
    text: str
    language: str = "en"
    voice: str = "alloy"
    group: Optional[str] = None

class BookmarkIn(BaseModel):
    client_id: str
    question: str
    answer: str
    language: str = "en"
    citations: List[dict] = []
    created_at: Optional[str] = None

class ChatFollowupIn(BaseModel):
    message: str
    answer: str
    language: str = "en"
    language_name: str = "English"

# ── Helper ────────────────────────────────────────────────────────────────────

def openai_client() -> AsyncOpenAI:
    base_url = "https://integrations.emergentagent.com/llm/openai/v1"
    return AsyncOpenAI(api_key=EMERGENT_LLM_KEY, base_url=base_url)

# ── Routes ────────────────────────────────────────────────────────────────────
@router.post("/client-error-log")
async def client_error_log(body: ClientErrorLogIn, user: dict = Depends(current_user)):
    await db.client_error_logs.insert_one({
        "id": str(uuid.uuid4()),
        "user_id": user["id"],
        "context": body.context[:100],
        "error_name": (body.error_name or "")[:200],
        "error_message": (body.error_message or "")[:1000],
        "platform": (body.platform or "")[:20],
        "retried": bool(body.retried),
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    return {"ok": True}

# ---------- Push notifications (Emergent-managed / SuprSend relay) ----------
# Registers this device's push token with the relay so it can be targeted by
# user_id later (daily nudge + dead-law bookmark alert — see push_jobs.py, or
# any future ad-hoc send). Never stores the raw device token in our own DB —
# the relay is the source of truth for token -> user_id mapping.
@router.get("/health/push")
async def health_push():
    """9-step deployment-readiness check for push notifications.

    Public because it never returns credentials — only booleans / short strings.
    Used by the deploy script + admin panel to verify Android push wiring
    end-to-end without needing to trigger a real send.
    """
    import os as _os
    push_key = _os.environ.get("EMERGENT_PUSH_KEY", "")
    key_present = bool(push_key) and push_key != "placeholder"
    return {
        "steps": {
            "1_android_package": "com.msafesolutions.legalaid",
            "2_google_services_file": "./google-services.json (see app.json android.googleServicesFile)",
            "3_notifications_plugin": "expo-notifications configured with default channel",
            "4_permission_declared": "android.permission.POST_NOTIFICATIONS",
            "5_backend_relay_module": "push.py (SuprSend via Emergent)",
            "6_emergent_push_key_present": key_present,
            "7_register_endpoint": "POST /api/register-push",
            "8_frontend_registers_native_token": "src/push.ts uses getDevicePushTokenAsync",
            "9_background_jobs_scheduled": "push_jobs.run_push_jobs_loop (daily 12:00 UTC)",
        },
        "gateway": gateway_status(),
        "note": (
            "EMERGENT_PUSH_KEY is replaced with the real value at deploy time. "
            "A `false` for step 6 in the dev pod is expected and does NOT block "
            "deployment — the deploy pipeline injects the real key."
        ),
        "ok": True,
    }


@router.post("/register-push")
async def register_push(body: RegisterPushBody, user: dict = Depends(current_user)):
    if body.user_id != user["id"]:
        raise HTTPException(403, "user_id mismatch")
    try:
        await register_device(user_id=body.user_id, platform=body.platform, device_token=body.device_token)
    except Exception as e:
        logger.warning(f"[push] register_push failed for user {user['id']}: {type(e).__name__}: {e}")
        # Never fail the caller's app flow over a push-registration hiccup —
        # the user just won't get push this session; nothing else breaks.
        return {"status": "failed"}
    return {"status": "registered"}


# One-tap self test: sends a push to the CALLING user's own device only.
# Rate-limited to 1 send per user per minute so a button-mash cannot spam the
# relay (or a legit user accidentally). Safe to expose to any authenticated
# caller — you cannot address someone else's device with this endpoint.
_PUSH_SELFTEST_COOLDOWN_S = 60
@router.post("/push/self-test")
async def push_self_test(user: dict = Depends(current_user)):
    from push import send_push as _send_push
    import time as _time
    now = _time.time()
    last = user.get("push_selftest_at_ts") or 0
    if now - float(last) < _PUSH_SELFTEST_COOLDOWN_S:
        remaining = int(_PUSH_SELFTEST_COOLDOWN_S - (now - float(last)))
        raise HTTPException(429, f"Please wait {remaining}s before sending another test.")
    try:
        await _send_push(
            recipients=[user["id"]],
            data={
                "title": "Dhara — test notification",
                "message": "If you see this, push notifications are working end-to-end. ✅",
                "action_url": "/(tabs)/settings",
            },
            idempotency_key=f"selftest:{user['id']}:{int(now)}",
        )
    except Exception as e:
        logger.warning(f"[push] self-test failed for user {user['id']}: {type(e).__name__}: {e}")
        return {"status": "failed", "reason": str(e)[:200]}
    await db.users.update_one(
        {"id": user["id"]}, {"$set": {"push_selftest_at_ts": now}},
    )
    return {"status": "sent"}

# ---------- Chat ----------
@router.post("/chat/stream")
async def chat_stream(body: ChatIn, user: dict = Depends(current_user)):
    session_id = body.session_id or str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    # DHARA-P0-4: immediate-danger assessment runs FIRST — before the guest
    # C1 counter, the SOS fast-path and every conversational shortcut below.
    _emergency = _is_immediate_danger(body.message)

    # -------- Guest Mode: 24h expiry + 5-question quota --------
    # Guests are real `users` docs (is_guest=True, minted by POST /api/auth/guest)
    # that flow through this ENTIRE pipeline unchanged below — this is the only
    # new block. Mode is pinned to "basic" so a guest never reaches the
    # Pro-paywall branch right after this.
    is_guest = bool(user.get("is_guest"))
    guest_question_count: Optional[int] = None
    guest_question_limit = int(user.get("guest_question_limit", 5)) if is_guest else None
    guest_warning = False
    if is_guest:
        expires_raw = user.get("guest_expires_at")
        try:
            expires_dt = datetime.fromisoformat(expires_raw) if expires_raw else None
        except Exception:
            expires_dt = None
        if expires_dt and datetime.now(timezone.utc) > expires_dt:
            # B-04b: safety before the expiry gate.
            _safety = _gate_safety_stream(
                body.message, body.language, _emergency,
                {
                    "type": "session", "session_id": session_id, "tier": "guest",
                    "mode": "basic", "sample_consumed": False,
                    "samples_remaining_after": 0, "guest": True,
                    "guest_expired": True,
                    "guest_question_count": int(user.get("guest_question_count", 0)),
                    "guest_question_limit": guest_question_limit,
                },
            )
            if _safety is not None:
                return _safety
            raise HTTPException(401, {
                "guest_expired": True,
                "message": "Your guest session has expired. Create a free account to keep using Dhara.",
            })
        current_count = int(user.get("guest_question_count", 0))
        if current_count >= guest_question_limit:
            def _guest_sse(obj: dict) -> bytes:
                return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8")

            async def _guest_limit_gen():
                yield _guest_sse({
                    "type": "session", "session_id": session_id, "tier": "guest", "mode": "basic",
                    "sample_consumed": False, "samples_remaining_after": 0,
                    "guest": True, "guest_question_count": guest_question_limit,
                    "guest_question_limit": guest_question_limit,
                })
                # DHARA-P0-4: safety before quota. A guest in immediate danger
                # (or sending a bare distress call) must still receive the 112
                # banner / SOS reply even after the free-question limit — the
                # quota wall must NEVER gate a life-safety response. This reply
                # is deterministic: it does not call the LLM and does not consume
                # a guest question.
                if _emergency:
                    _banner = _EMERGENCY_BANNER.get(body.language, _EMERGENCY_BANNER["en"])
                    yield _guest_sse({"type": "delta", "content": _banner, "emergency": True})
                elif FEATURE_C1_CONVERSATIONAL_REPLIES and is_bare_distress(body.message):
                    _sos = (
                        get_conversational_template("distress_sos", body.language)
                        or _EMERGENCY_BANNER.get(body.language, _EMERGENCY_BANNER["en"])
                    )
                    yield _guest_sse({"type": "delta", "content": _sos, "c1_sos": True})
                else:
                    yield _guest_sse({
                        "type": "guest_limit_reached",
                        "message": "You've reached the guest limit. Create a free account to continue.",
                    })
                yield _guest_sse({"type": "done"})
            return StreamingResponse(
                _guest_limit_gen(), media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        # C1 guard: conversational greetings / capability queries are free —
        # a guest saying "hi" or "thanks" should not consume a free question.
        # The deny-list check runs first (deny always wins over the allowlist),
        # so "help me" or "danger" messages are never treated as free C1 turns.
        # Exception: bare distress phrases ("help", "bachao") DO bypass the
        # deny-list and return the caring SOS reply — they must also not
        # consume a guest question.
        _c1_pre = (
            FEATURE_C1_CONVERSATIONAL_REPLIES
            and not _emergency
            and (
                is_bare_distress(body.message)
                or (
                    not is_denylisted(body.message)
                    and bool(classify_conversational_intent(body.message, body.language))
                )
            )
        )
        if not _c1_pre:
            updated = await db.users.find_one_and_update(
                {"id": user["id"]},
                {"$inc": {"guest_question_count": 1}},
                return_document=True,
            )
            guest_question_count = int((updated or {}).get("guest_question_count", current_count + 1))
            guest_warning = guest_question_count == guest_question_limit - 1
        else:
            # C1 turn: report the current count unchanged (no increment).
            guest_question_count = current_count
            guest_warning = False

    # -------- Paywall enforcement for Pro-mode capabilities --------
    # Basic mode → always allowed, unlimited.
    # Pro mode + is_pro → allowed, no counter.
    # Pro mode + not is_pro + samples_used < N → allowed as SAMPLE (increment counter).
    # Pro mode + not is_pro + samples_used >= N → HTTP 402 with paywall payload.
    # Guests are always pinned to basic mode — they never see the Pro paywall.
    mode = "basic" if is_guest else (body.mode or "basic").lower()
    is_pro_user = bool(user.get("is_pro"))
    samples_used = int(user.get("pro_samples_used", 0))
    is_sample_consumption = False
    # Advocate mode: professional prompt for licensed advocates — no paywall
    is_advocate_mode = (mode == "advocate")

    if mode == "pro" and not is_pro_user and not is_advocate_mode:
        if samples_used >= PRO_FREE_SAMPLES:
            # B-04c: safety before the paywall gate.
            _safety = _gate_safety_stream(
                body.message, body.language, _emergency,
                {
                    "type": "session", "session_id": session_id,
                    "tier": "pro" if is_pro_user else "free", "mode": mode,
                    "sample_consumed": False, "samples_remaining_after": 0,
                    "guest": False, "paywall_blocked": True,
                },
            )
            if _safety is not None:
                return _safety
            # Log the blocked attempt so the operator sees who's hitting the paywall
            # and what they were trying to ask. Otherwise the query vanishes from the DB.
            try:
                await db.messages.insert_one({
                    "id": str(uuid.uuid4()),
                    "session_id": session_id,
                    "user_id": user["id"],
                    "role": "user",
                    "content": body.message,
                    "language": body.language,
                    "mode": mode,
                    "status": "blocked_paywall",
                    "created_at": now,
                    "timestamp": now,
                })
            except Exception:
                logger.exception("Failed to log paywall-blocked query")
            raise HTTPException(
                status_code=402,
                detail={
                    "paywall": True,
                    "reason": "pro_samples_exhausted",
                    "samples_used": samples_used,
                    "samples_limit": PRO_FREE_SAMPLES,
                    "pro_price_inr_paise": PRO_PRICE_INR,
                    "pro_price_label": PRO_PRICE_LABEL,
                    "pro_price_usd_cents": PRO_PRICE_USD,
                    "pro_price_usd_label": PRO_PRICE_USD_LABEL,
                    "message": (
                        f"You've used all {PRO_FREE_SAMPLES} free Pro-quality samples. "
                        "Upgrade to Pro to unlock unlimited lawyer-style deep answers, drafts, "
                        "action plans and escalation paths."
                    ),
                },
            )
        is_sample_consumption = True

    session = await db.sessions.find_one({"id": session_id, "user_id": user["id"]})
    if not session:
        title = body.message[:60]
        await db.sessions.insert_one({
            "id": session_id,
            "user_id": user["id"],
            "title": title,
            "language": body.language,
            "mode": mode,
            "tier": "pro" if is_pro_user else "free",
            "created_at": now,
            "updated_at": now,
        })

    await db.messages.insert_one({
        "id": str(uuid.uuid4()),
        "session_id": session_id,
        "user_id": user["id"],
        "role": "user",
        "content": body.message,
        "language": body.language,
        "mode": mode,
        "status": "ok",  # updated below if refusal fires
        "created_at": now,
        "timestamp": now,  # explicit alias for CSV clarity
    })

    # ── SOS fast-path: bypass the ~2.5 s LLM+corpus call entirely ──────────
    # is_bare_distress matches ONLY very short bare phrases ("help", "help me",
    # "bachao", "মদ্দ করুন") — longer messages with legal context go through
    # the normal pipeline unchanged.  At this point mode / tier / guest data
    # are all settled so we can emit a complete SSE session frame immediately.
    if FEATURE_C1_CONVERSATIONAL_REPLIES and not _emergency and is_bare_distress(body.message):
        _sos_reply = get_conversational_template("distress_sos", body.language)
        if _sos_reply:
            def _sse_sos(obj: dict) -> bytes:
                return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8")

            async def _sos_fast_gen():
                yield _sse_sos({
                    "type": "session",
                    "session_id": session_id,
                    "tier": "guest" if is_guest else ("pro" if is_pro_user else "free"),
                    "mode": mode,
                    "sample_consumed": False,
                    "samples_remaining_after": max(0, PRO_FREE_SAMPLES - samples_used),
                    "guest": is_guest,
                    "guest_question_count": guest_question_count,
                    "guest_question_limit": guest_question_limit,
                    "guest_warning": guest_warning,
                })
                # c1_sos flag tells the frontend to show the persistent emergency banner
                yield _sse_sos({"type": "delta", "content": _sos_reply, "c1_sos": True})
                _ts = datetime.now(timezone.utc).isoformat()
                await db.messages.insert_one({
                    "id": str(uuid.uuid4()),
                    "session_id": session_id,
                    "user_id": user["id"],
                    "role": "assistant",
                    "content": _sos_reply,
                    "language": body.language,
                    "mode": mode,
                    "model_provider": "dhara_c1",
                    "model_name": "distress_sos",
                    "tier": "guest" if is_guest else ("pro" if is_pro_user else "free"),
                    "sample_consumed": False,
                    "status": "ok",
                    "created_at": _ts,
                    "timestamp": _ts,
                })
                await db.sessions.update_one(
                    {"id": session_id}, {"$set": {"updated_at": _ts}}
                )
                yield _sse_sos({"type": "done"})

            return StreamingResponse(
                _sos_fast_gen(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

    # -------- Gate 2: Pre-retrieval pipeline (A → B → C) --------
    _current_session = await db.sessions.find_one({"id": session_id, "user_id": user["id"]})
    _v2_lq, _v2_classify, _v2_fact, _v2_state = await run_pre_retrieval(
        query=body.message,
        language=body.language,
        llm_key=EMERGENT_LLM_KEY,
        session=_current_session,
        db=db,
    )
    # If CLARIFICATION_ANSWER, the pipeline restores the original query in lq.normalized_query.
    # Use that as the retrieval text so we retrieve on the original issue.
    _retrieval_override = (
        _v2_lq.normalized_query
        if (_v2_lq.normalized_query and _v2_lq.normalized_query != body.message)
        else None
    )
    # P0-D: if this turn's own text has no explicit Act+section AND is not
    # already being overridden by a restored clarification-answer query, but
    # an explicit statute reference IS active from an earlier turn (P0-C's
    # CaseState.active_statute_ref), carry it forward into retrieval so a
    # plain follow-up phrased as a longer continuation (not a short
    # clarification) still retrieves against the right section.
    if not _retrieval_override and _v2_state and _v2_state.active_statute_ref:
        _own_act, _own_sec, _own_sub = _p0d_extract_exact_ref(body.message)
        if not (_own_act and _own_sec):
            _ref = _v2_state.active_statute_ref
            _retrieval_override = f"{body.message} ({_ref.get('act')} Section {_ref.get('section')})"
    # If Layer C says we need a material clarification question, emit it and stop.
    if _v2_fact.answer_mode == "ESCALATE" and _v2_fact.follow_up_question:
        async def _clarify_gen():
            yield f"data: {json.dumps({'type': 'session', 'session_id': session_id, 'tier': 'guest' if is_guest else ('pro' if is_pro_user else 'free'), 'mode': mode, 'sample_consumed': False, 'samples_remaining_after': PRO_FREE_SAMPLES - samples_used, 'guest': is_guest, 'guest_question_count': guest_question_count, 'guest_question_limit': guest_question_limit, 'guest_warning': guest_warning}, ensure_ascii=False)}\n\n".encode()
            q = _v2_fact.follow_up_question
            yield f"data: {json.dumps({'type': 'delta', 'content': q}, ensure_ascii=False)}\n\n".encode()
            await db.messages.insert_one({"id": str(uuid.uuid4()), "session_id": session_id, "user_id": user["id"], "role": "assistant", "content": q, "language": body.language, "mode": mode, "status": "clarification_asked", "created_at": datetime.now(timezone.utc).isoformat(), "timestamp": datetime.now(timezone.utc).isoformat()})
            await save_case_state(db, session_id, _v2_state)
            if is_guest and guest_question_count == guest_question_limit:
                _limit_msg = "You've reached the guest limit. Create a free account to continue."
                yield f"data: {json.dumps({'type': 'guest_limit_reached', 'message': _limit_msg}, ensure_ascii=False)}\n\n".encode()
            yield f"data: {json.dumps({'type': 'done'})}\n\n".encode()
        return StreamingResponse(_clarify_gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # -------- Retrieval-language bridge (Hindi/Indic search fix) --------
    # Use the pipeline's restored normalized_query for clarification turns
    _base_retrieval_text = _retrieval_override or body.message
    retrieval_text = _base_retrieval_text
    _is_indic = needs_retrieval_translation(_base_retrieval_text)
    if _is_indic:
        retrieval_text = await translate_for_retrieval(_base_retrieval_text)
        _same = retrieval_text == _base_retrieval_text
        logger.info(
            "retrieval-bridge | indic=True | translation_ok=%s | q=%r | t=%r",
            not _same, body.message[:80], retrieval_text[:80],
        )
    else:
        logger.info("retrieval-bridge | indic=False | q=%r", body.message[:80])

    # -------- Retrieval + citation integrity (P1) --------
    # (a) Non-legal / non-Indian jurisdiction → hard refusal, no LLM call
    early_refusal: Optional[str] = None
    if is_non_indian_jurisdiction(retrieval_text):
        early_refusal = REFUSAL_NON_INDIAN
    # Check the user's RAW message as well as the normalised text: the LLM
    # query-normaliser often rephrases "should I forgive my husband" into a
    # neutral legal-sounding question, which silently slips past this
    # keyword gate and sends a request for personal/moral advice into the
    # legal corpus. (Same class of bug as the C1 vocative-stripper fix.)
    elif is_non_legal_advice(retrieval_text) or is_non_legal_advice(body.message):
        early_refusal = REFUSAL_NOT_LEGAL

    # -------- Personal-law disambiguation (marriage/succession) --------
    # Hindu Marriage Act vs Special Marriage Act vs Muslim/Christian personal
    # law (same for succession) give DIFFERENT sections for the same real-life
    # question — answering under the wrong one is a wrong citation. Ask once
    # (remembered for the rest of THIS chat thread) instead of guessing; see
    # personal_law.py for the full design note.
    personal_law_topic: Optional[str] = None
    personal_law_note: Optional[str] = None
    if not early_refusal:
        personal_law_topic = classify_personal_law_topic(retrieval_text)
        if personal_law_topic:
            personal_law_ctx = detect_personal_law_context(retrieval_text) or (
                (session or {}).get("personal_law_context") or {}
            ).get(personal_law_topic)
            if not personal_law_ctx:
                early_refusal = disambiguation_question(personal_law_topic, body.language)
            else:
                await db.sessions.update_one(
                    {"id": session_id},
                    {"$set": {f"personal_law_context.{personal_law_topic}": personal_law_ctx}},
                )
                hint = act_hint_phrase(personal_law_ctx, personal_law_topic)
                if hint:
                    retrieval_text = f"{retrieval_text} ({hint})"
                personal_law_note = act_disclaimer(personal_law_ctx, personal_law_topic, body.language)

    # ── Property-domain hint ────────────────────────────────────────────────
    # Generic property queries ("property rights", "land dispute", "RERA
    # complaint") have low corpus recall because the embedding needs specific
    # act names to match.  Append act hints when the query looks like a
    # property topic but doesn't already name a relevant statute.
    # IMPORTANT: use whole-word matching (\b) to avoid matching sub-strings
    # like "rent" inside "parents" or "transfer" inside "transferred".
    if not early_refusal:
        _PROP_TRIGGERS_RE = re.compile(
            r"\b(?:property|land|rera|real\s+estate|lease|tenant|rent(?:al)?|eviction|"
            r"possession|registration|stamp\s+duty|mutation|ownership|title(?:\s+deed)?|"
            r"easement|mortgage|flat|builder|plot|landlord|occupancy|trespass)\b",
            re.IGNORECASE,
        )
        _PROP_ACTS = (
            "transfer of property act", "rera", "real estate regulation",
            "registration act", "rent control", "land acquisition",
        )
        _rt_lower = (retrieval_text or "").lower()
        _has_prop_trigger = bool(_PROP_TRIGGERS_RE.search(retrieval_text or ""))
        _has_prop_act = any(a in _rt_lower for a in _PROP_ACTS)
        if _has_prop_trigger and not _has_prop_act:
            retrieval_text = (
                f"{retrieval_text} "
                "(Transfer of Property Act 1882, RERA 2016, Registration Act 1908)"
            )
            logger.info("property-hint injected | q=%r", body.message[:80])

    # (b) Corpus retrieval — deterministic keyword match against verified statutes
    retrieved = [] if early_refusal else corpus_retrieve(retrieval_text, limit=3)

    # (b-own) RTI-"ignored" fix: for a NEW query (not a clarification-answer
    # restore) the LLM query-parser's rewritten normalized_query can drop the
    # user's own decisive words ("complain", "ignored"), so the verified
    # complaint/appeal sections (RTI 18/19) lost to generic RTI 6/3. Keep the
    # user's own top hits FIRST, then the rewrite's, capped at three.
    _last_turn = ((_v2_state.turn_type_history or [None])[-1]) if _v2_state else None
    if (
        retrieved is not None and not early_refusal and not _is_indic
        and _retrieval_override and _retrieval_override == _v2_lq.normalized_query
        and _last_turn != "CLARIFICATION_ANSWER"
    ):
        _own_hits = corpus_retrieve(body.message, limit=3)
        _own_keys = {it.get("key") for it in _own_hits}
        retrieved = (_own_hits + [it for it in retrieved if it.get("key") not in _own_keys])[:3]

    # (b0) State / UT layer — rent, liquor, traffic compounding and stamp duty are
    # state subjects. If the user has told us their state we serve its verified
    # rules ALONGSIDE the central law; we never substitute another state's rule.
    # AQF Phase 1b: when FEATURE_MATTER_CONTEXT_V2 is ON, resolve_jurisdiction_state
    # unifies the session-level jurisdiction (from CaseState.legal_query) with the
    # user-profile state, giving in-session confirmation precedence.
    if FEATURE_MATTER_CONTEXT_V2:
        _session_jur_state = (
            (
                _v2_state.legal_query.jurisdiction.state
                if _v2_state.legal_query and _v2_state.legal_query.jurisdiction
                else None
            )
            or None
        )
        _resolved = resolve_jurisdiction_state(_session_jur_state, user.get("state"))
        user_state = (_resolved["state"] or "").upper()
    else:
        user_state = (user.get("state") or "").upper()
    state_hits = [] if early_refusal else corpus_retrieve_state(retrieval_text, user_state, limit=2)
    state_topic = None if early_refusal else state_sensitive_topic(retrieval_text)

    # (b1) Citation integrity: if the user's query explicitly names a section/article
    # identifier (e.g. "BNS Section 999", "Article 350", "BNSS 220") and NONE of the
    # retrieved entries actually match that identifier, void the retrieval. Otherwise
    # unrelated but keyword-adjacent chips would appear alongside a refusal, which is
    # dangerous and contradicts the P1 accuracy rule.
    if retrieved and not early_refusal:
        import re as _re_id
        def _norm_short(s: str) -> str:
            return _re_id.sub(r"[^a-z0-9()]+", " ", (s or "").lower()).strip()
        query_ids: list[str] = []
        # "Article 21", "Article 21A"
        for m in _re_id.finditer(r"\b[Aa]rticle\s+(\d+[A-Za-z]?)\b", retrieval_text):
            query_ids.append(f"article {m.group(1).lower()}")
        # "BNS Section 999", "BNSS Section 43(5)", "Sec. 43 BNSS", "BNSS 43(5)",
        # "RTI Section 6", "CPA 34", "MV 185", "Motor Vehicles Section 185"
        # P1 dual-corpus fix: expanded whitelist to cover more commonly-cited
        # Act abbreviations (IT, NI, IBC, HMA, HSA, SMA, POCSO, NDPS, PCMA).
        for m in _re_id.finditer(
            r"\b(BNS|BNSS|BSA|IPC|CrPC|PWDVA|RTI|CPA|MV|MVA|IT|NI|IBC|HMA|HSA|SMA|POCSO|NDPS|PCMA|CAL|SWPSC|IR|SS|DV)\b[^\w]*(?:Sec(?:tion|\.)?\s*)?(\d+[A-Za-z]?(?:\(\d+\))?)\b",
            retrieval_text, _re_id.IGNORECASE,
        ):
            query_ids.append(f"{m.group(1).lower()} {m.group(2).lower()}")
        if query_ids:
            hit_labels = { _norm_short(it["short_label"]) for it in retrieved }
            if not any(qid in hit_labels or any(qid in hl for hl in hit_labels) for qid in query_ids):
                # Query names an identifier we don't cover → void retrieval so user sees refusal only
                retrieved = []

    # (b2) MongoDB corpus — 70,397 sections across CENTRAL + 12 states.
    # Runs ALONGSIDE the Python corpus, not instead of it. Python corpus
    # contributes scope_notes and citizen-language explanations; MongoDB
    # contributes verbatim government text and breadth (2,120 Acts).
    # Three safety guards enforced in code (not prompts):
    #   G1 — dead_warning  : serve_warning prepended for is_dead_law sections
    #   G2 — judicial_flag : user_warning from judicial_invalidations (status != UPHELD)
    #   G3 — badge         : verify_tier==1 + verified_by → "Advocate Reviewed"
    db_hits: list[dict] = []
    if not early_refusal:
        try:
            db_hits = await db_retrieve(corpus_db, retrieval_text, state_code=user_state or None, limit=3)
        except Exception as e:
            db_hits = []   # MongoDB unavailable — Python corpus handles it
            logger.warning("[chat] db_retrieve failed (falling back to Python corpus only): %s", e)

    # (b3) Cross-corpus citation-conflict guard. Logged issue, now fixed:
    # querying "Information Technology Act section 66A" surfaced the hand-
    # curated Python-corpus entry "IT 43" FIRST (it scores on the generic
    # phrase "information technology act", which the query obviously
    # contains), ahead of MongoDB's own exact match for the section the user
    # actually asked about. The (b1) integrity check above only recognises a
    # closed whitelist of Act prefixes (BNS, IPC, MV, ...) and never covers
    # this case since "Information Technology" isn't one of them.
    #
    # Generalised fix: whenever the query cites an explicit section number
    # AND MongoDB found an EXACT match for that exact number, any Python-
    # corpus hit whose OWN section number (parsed off the end of its
    # short_label, e.g. "43" in "IT 43") is a DIFFERENT number is dropped —
    # the user asked about one specific section, not "something in this Act
    # family". Hits with no trailing number (e.g. "Cyber report 1930" is a
    # helpline, not a section) are left untouched.
    if retrieved and db_hits and not early_refusal:
        _cited_sec_match = re.search(
            r"(?:section|sections|sec|s\.|art(?:icle)?\.?)\s*(\d+[A-Za-z]{0,3})",
            retrieval_text, re.IGNORECASE,
        )
        if _cited_sec_match:
            _cited_sec = _cited_sec_match.group(1).upper()
            if any((h.get("section_number") or "").upper() == _cited_sec for h in db_hits):
                def _own_section_number(short_label: str) -> Optional[str]:
                    m = re.search(r"(\d+[A-Za-z]{0,3})$", short_label or "")
                    return m.group(1).upper() if m else None
                retrieved = [
                    it for it in retrieved
                    if _own_section_number(it.get("short_label", "")) in (None, _cited_sec)
                ]

    # (b3b) P1 grounding-safety fix for the BNS 199 / IPC 166A contradiction:
    # when an EXPLICIT statute reference is active for this turn (this turn's
    # own citation, or one carried forward via _v2_state.active_statute_ref)
    # but the exact MongoDB lookup found NOTHING (db_hits empty), a Python-
    # corpus hit that is NOT itself about that exact section (sharing only
    # generic words with the query, e.g. a completely unrelated section)
    # cannot be trusted to answer this specific-section question — drop it.
    #
    # P1 follow-up (citation-guard parity fix): this must NOT also drop a
    # Python-corpus hit whose OWN section number IS the cited one — e.g.
    # corpus_ipc.py's hand-curated "ipc_34" entry answering "what does IPC
    # Section 34 say" is an exact, deliberate match, not fuzzy noise, and
    # was being wrongly wiped here before Layer S ever got a chance to
    # apply its historical-still-applicable framing. Reuses the SAME
    # own-section-number parsing as guard (b3) just above.
    # Reuses the SAME active-reference signal as (b3)/P0-D — no new engine.
    _active_ref_now = _v2_state.active_statute_ref if _v2_state else None
    if not _active_ref_now:
        _this_turn_act, _this_turn_sec, _ = _p0d_extract_exact_ref(body.message)
        if _this_turn_act and _this_turn_sec:
            _active_ref_now = {"act": _this_turn_act, "section": _this_turn_sec}
            # P0 fix: store on _v2_state immediately (same dict object, not a
            # copy) so any later mutation of _active_ref_now (e.g. the
            # dead-law-with-successor "sticky" marker below) is actually
            # persisted by save_case_state() for the NEXT turn — otherwise a
            # follow-up like "what is the BNS equivalent?" loses the fact
            # that the conversation is about a verified dead-law citation.
            if _v2_state is not None:
                _v2_state.active_statute_ref = _active_ref_now
    if _active_ref_now and not db_hits and retrieved and not early_refusal:
        def _own_section_number_b3b(entry: dict) -> Optional[str]:
            m = re.search(r"(\d+[A-Za-z]{0,3})$", entry.get("short_label", "") or "")
            if m:
                return m.group(1).upper()
            km = re.search(r"_(\d+[A-Za-z]{0,3})$", (entry.get("key") or ""))
            return km.group(1).upper() if km else None
        # P0 fix (Act+Section identity): section number alone is not a unique
        # key — "IPC 103" and "BNS 103" share the same number but are
        # different provisions. Require the ENTRY's own Act to also match
        # the active reference's Act, not just the trailing number.
        _ACT_PREFIX_CANON = {
            "ipc": "indian penal code", "bns": "bharatiya nyaya sanhita",
            "bnss": "bharatiya nagarik suraksha sanhita",
            "bsa": "bharatiya sakshya adhiniyam",
            "crpc": "code of criminal procedure",
        }
        def _own_act_matches_b3b(entry: dict, ref_act_lower: str) -> bool:
            if not ref_act_lower:
                return True
            entry_act = (entry.get("act_name") or entry.get("act") or "").strip().lower()
            if entry_act:
                return entry_act == ref_act_lower
            label = (entry.get("short_label") or "").strip()
            prefix = label.split()[0].lower() if label.split() else ""
            return _ACT_PREFIX_CANON.get(prefix) == ref_act_lower
        _ref_sec_b3b = (_active_ref_now.get("section") or "").upper()
        _ref_act_b3b = (_active_ref_now.get("act") or "").strip().lower()
        _before_b3b = len(retrieved)
        retrieved = [
            it for it in retrieved
            if _own_section_number_b3b(it) == _ref_sec_b3b and _own_act_matches_b3b(it, _ref_act_b3b)
        ]
        if _before_b3b > len(retrieved):
            logger.info(
                "[grounding_guard] dropping %d unverified fuzzy hit(s) for %s %s — "
                "not an exact match for the active reference's own section",
                _before_b3b - len(retrieved), _active_ref_now.get("act"), _active_ref_now.get("section"),
            )

    # Orphan-invalidation check — sections in judicial_invalidations but NOT
    # in legal_sections (e.g. IPC §377: IPC replaced by BNS, absent from corpus
    # but Supreme Court ruling is tracked).  Fires only when db_hits is empty.
    db_orphan: dict | None = None
    if not early_refusal and not db_hits:
        try:
            db_orphan = await db_orphan_check(corpus_db, retrieval_text)
        except Exception as e:
            db_orphan = None
            logger.warning("[chat] db_orphan_check failed: %s", e)

    # (c) If no retrieval hit AND no early refusal, we still refuse (no verified source).
    # db_hits = MongoDB verbatim sections; db_orphan = standalone judicial ruling
    # for an act not in the corpus.  Either of these counts as a verified source.
    if not early_refusal and not retrieved and not state_hits and not db_hits and not db_orphan:
        early_refusal = REFUSAL_NO_CORPUS
        # P1 fix: a missing legal_sections record does NOT mean the section
        # doesn't exist — check the verified IPC<->BNS correspondence index
        # before saying anything more specific than the generic refusal.
        # Never swap the user's own Act for another; only ever OFFER a
        # hedged hypothesis when the number matches a former IPC/CrPC/
        # Evidence-Act number, and always say the limitation out loud.
        _ref_for_hint = _active_ref_now
        _hint_act = (_ref_for_hint or {}).get("act") or ""
        _hint_sec = (_ref_for_hint or {}).get("section") or ""
        if _ref_for_hint and re.search(r"bharatiya|bns|bnss|bsa", _hint_act, re.IGNORECASE):
            try:
                _existence = db_check_bns_family_identifier(_hint_sec)
            except Exception as e:
                _existence = {"confirmed_existing": False, "ipc_origin_hypothesis": None}
                logger.warning("[grounding_guard] check_bns_family_identifier failed: %s", e)
            if not _existence.get("confirmed_existing") and _existence.get("ipc_origin_hypothesis"):
                early_refusal = build_bns_identifier_hypothesis_response(
                    _hint_act, _hint_sec, _existence["ipc_origin_hypothesis"],
                )
            # confirmed_existing=True: leave early_refusal as the plain
            # REFUSAL_NO_CORPUS — this is an honest "missing grounding" gap
            # for a real section, not a naming mix-up.

    logger.info(
        "retrieval-result | corpus=%d | state=%d | db=%d | orphan=%s | branch=%s",
        len(retrieved), len(state_hits), len(db_hits),
        bool(db_orphan),
        "refusal" if early_refusal else "rag",
    )

    # (c0) Cross-engine noise suppression (general rule, not per-Act — lives
    # here once, not duplicated per topic). The curated Python corpus
    # (`retrieved`) is hand-verified: every entry was deliberately keyword-
    # tagged for that exact scenario. MongoDB's fuzzy $text search, in
    # contrast, can pass a hit on nothing more than two generic words ("day",
    # "every") shared with a completely unrelated Act — e.g. "my husband
    # beats me every day" once also surfaced "The Representation of the
    # People Act... paid holiday to employees on the day of poll" (shared
    # words: "day", "every"), and "can an Indian marry twice" surfaced "The
    # National Commission for Indian System of Medicine Act" (shared word:
    # "indian"). `is_anchored` (corpus_db.py) marks a MongoDB hit as
    # STRUCTURALLY trustworthy — an exact section-number lookup, or an
    # explicit Act-label match — as opposed to bare generic-word overlap.
    # Once ANY verified answer already exists for this query (the curated
    # corpus answered, OR MongoDB itself found an anchored hit), a
    # non-anchored MongoDB hit adds noise, not evidence, and is dropped. It
    # is kept ONLY when it is the sole thing found for the query at all —
    # never leaving the user with a refusal just because the one hit we have
    # happens to be unanchored.
    if db_hits and (retrieved or any(h.get("is_anchored") for h in db_hits)):
        db_hits = [h for h in db_hits if h.get("is_anchored")]

    # (c0b) Cross-corpus deduplication — P1 dual-corpus fix.
    # When a section appears in BOTH the Python corpus (citizen-language) and
    # MongoDB (verbatim government text), the user sees duplicate context and
    # the LLM may produce contradictory sentences. Deduplicate by matching
    # on section number + act family:
    #   - If MongoDB has an anchored hit for the same section, keep MongoDB
    #     (it has verbatim text + live safety guards G1/G2/G3) and drop the
    #     Python corpus duplicate — the LLM already receives the scope_note
    #     from the system prompt's persona, not from the corpus context.
    #   - If only the Python corpus has it (MongoDB missed it or was generic),
    #     the Python entry stays untouched.
    if retrieved and db_hits:
        _db_section_keys: set[str] = set()
        for h in db_hits:
            act = (h.get("act_name") or "").lower()
            sec = (h.get("section_number") or "").upper()
            if sec:
                _db_section_keys.add(f"{sec}")
                # Also index with abbreviated act name for cross-matching.
                # Keep this list in sync with the (b1) whitelist above.
                for prefix in (
                    "bns", "bnss", "bsa", "pwdva", "rti", "cpa",
                    "it", "ni", "mv", "mva", "ibc", "hma", "hsa", "sma",
                    "pocso", "ndps", "pcma", "ipc", "crpc",  # P1 — aligned with (b1) whitelist
                ):
                    if prefix in act:
                        _db_section_keys.add(f"{prefix}_{sec}")
        def _python_entry_duplicated_in_db(entry: dict) -> bool:
            label = (entry.get("short_label") or "").strip()
            if not label:
                return False
            # Parse "BNS 189" → prefix="bns", sec="189"
            parts = label.split()
            if len(parts) >= 2:
                prefix = parts[0].lower()
                sec = parts[-1].upper()
                # Bug fix (P1 dual-corpus): previously also checked `sec in _db_section_keys`
                # (bare section number, no act context). That caused IT Act Section 43 to
                # be dropped whenever MongoDB returned Section 43 from ANY unrelated act
                # (e.g. Mines Act, Labour Act). Now we only match on act-aware keys
                # (prefix_sec) so the dedup is scoped to the same act family.
                if f"{prefix}_{sec}" in _db_section_keys:
                    return True
            return False
        before_count = len(retrieved)
        retrieved = [it for it in retrieved if not _python_entry_duplicated_in_db(it)]
        if before_count > len(retrieved):
            logger.info("[dual-corpus-dedup] dropped %d Python corpus entries duplicated in MongoDB", before_count - len(retrieved))

    # (c1) Merge the state rules into the citation list the user will see. State
    # rules go LAST so the central position is read first, and the combined list
    # stays capped at three chips.
    if state_hits:
        keep_central = max(0, 3 - len(state_hits))
        retrieved = retrieved[:keep_central] + state_hits

    # -------- Gate 2: Post-retrieval pipeline (H → S) --------
    # Layer H: Irrelevance filter — hard guards (mosque→PoW Act, IPC→BNS, Designs→SIM)
    # Layer S: Law Status Guard — annotates SUPERSEDED/REPEALED from registry JSON
    retrieved, db_hits = run_post_retrieval(
        corpus_hits=retrieved,
        db_hits=db_hits,
        lq=_v2_lq,
        classify_result=_v2_classify,
    )

    # -------- P0 containment (T04/T05 ordering fix) -----------------------
    # Recheck successor-release eligibility ONCE per turn, on this turn's
    # full PRE-filter evidence, BEFORE either consumer reads it: (1) the
    # citation-truth-guard's own "keep genuine successor-act citations"
    # inclusion check just below, and (2) the Layer M cannot-verify bypass
    # further down. Previously the citation-truth-guard ran first and read
    # the `dead_with_successor` flag as it stood at the END of the PREVIOUS
    # turn (before this turn's recheck had run), while the Layer M bypass
    # used the freshly-rechecked value — the two consumers could disagree
    # about the same turn. Computing it once, here, first, and reusing the
    # result for both closes that ordering gap; mutates `_active_ref_now`
    # in place exactly as before, so the next turn's carry-forward is
    # unaffected.
    _has_explicit_dead_citation_with_successor = recheck_dead_with_successor_release(
        _active_ref_now, list(retrieved) + list(db_hits), _LEGACY_TO_SUCCESSOR_FAMILY,
    )

    # -------- P0-D: Citation Truth Guard — content validation ------------
    # Runs HERE, upstream of ALL downstream citation-payload construction AND
    # upstream of the CITATION_GUARD_ENABLED buffering/timing branch further
    # below — so this content check applies IDENTICALLY regardless of
    # streaming/buffering mode (the prior audit's finding: that flag only
    # ever controlled TIMING, never content). An Act-name match alone (the
    # corroboration_score() 9999 sentinel in corpus_db.py) is NOT sufficient
    # evidence that a db_hit supports the SPECIFIC exact section the active
    # conversation context is about — only a genuine exact lookup for this
    # turn (exact_identifier_requested=True) or an exact act+section match
    # against the active reference is trusted. Scope: only engages when an
    # explicit statute reference is actually active (P0-C); ordinary
    # non-citation conversations are completely unaffected.
    _p0d_active_ref = _v2_state.active_statute_ref if _v2_state else None
    if _p0d_active_ref and db_hits:
        _p0d_req_act = _p0d_active_ref.get("act")
        _p0d_req_sec = (_p0d_active_ref.get("section") or "").upper()
        def _p0d_hit_supports_active_ref(h: dict) -> bool:
            if h.get("exact_identifier_requested"):
                return True
            if (
                h.get("act_name") == _p0d_req_act
                and (h.get("section_number") or "").upper() == _p0d_req_sec
            ):
                return True
            # P0 fix (multi-turn follow-up): once this conversation's active
            # reference has been established as a dead law with a KNOWN
            # successor act family (see the "sticky" dead_with_successor
            # marker set above), a legitimately-retrieved hit from THAT
            # successor act also supports the active reference — e.g. after
            # "IPC Section 166A" a follow-up "what is the BNS equivalent?"
            # should be allowed to keep its genuine BNS hits, not have them
            # all stripped for not literally being IPC 166A.
            if _p0d_active_ref.get("dead_with_successor") and h.get("act_name") == _p0d_active_ref.get("successor_act_family"):
                return True
            return False
        _p0d_before = len(db_hits)
        db_hits = [h for h in db_hits if _p0d_hit_supports_active_ref(h)]
        if _p0d_before > len(db_hits):
            logger.info(
                "[citation_truth_guard] dropped %d unsupported same-Act citation(s) for active ref %s %s",
                _p0d_before - len(db_hits), _p0d_req_act, _p0d_req_sec,
            )

    # Layer M gate: if EVERY surviving provision is dead/superseded and there
    # is no live orphan judicial hit either, emit a deterministic cannot-verify
    # response instead of asking the LLM to describe a dead law in citizen
    # words. Refusal analytics still fires so we track how often this happens.
    #
    # P0 fix: this gate must NOT swallow an EXPLICIT exact-section lookup
    # (e.g. "IPC Section 166A") that has a verified, known successor
    # mapping (e.g. BNS Section 199) into a blanket "cannot verify"
    # response with zero citation. The user asked about THIS specific
    # section by name; the honest, more informative answer is the normal
    # citation path below — which already shows the verbatim text, a
    # "SAFETY WARNING (DEAD LAW)" annotation, and the successor act — not a
    # template that discards the citation entirely. The gate still fires
    # normally for any dead hit that was only a coincidental/fuzzy match on
    # a general query (no explicit citation), or that has no known
    # successor (a genuine unverifiable gap).
    # `_has_explicit_dead_citation_with_successor` was computed ONCE, earlier
    # in this function (right after run_post_retrieval, on pre-filter
    # evidence) — see the comment block there for why (P0 ordering fix,
    # T04/T05: the citation-truth-guard's own successor-family inclusion
    # check, just above, must see the SAME freshly-rechecked value, not a
    # later recomputation on already-filtered hits).
    _all_hits_for_status = list(retrieved) + list(db_hits)
    # Reuses the value already computed above (pre-filter) — NOT
    # recomputed here, so the citation-truth-guard filter above and this
    # Layer M bypass are guaranteed to agree for this turn.
    _cannot_verify = (
        not early_refusal
        and _all_hits_for_status
        and all_provisions_dead(_all_hits_for_status)
        and not db_orphan
        and not _has_explicit_dead_citation_with_successor
    )
    if _cannot_verify:
        # Convert into the same refusal code path so analytics + sample-debit
        # logic below both see it as a controlled refusal, not an LLM failure.
        early_refusal = build_cannot_verify_response(
            _v2_lq, body.language, dead_hits=_all_hits_for_status,
        )
        logger.info("[layer_m] cannot_verify triggered — %d dead provision(s)", len(_all_hits_for_status))

    # -------- Refusal analytics (A4 instrumentation) --------
    # Anonymized, aggregate-only event: NO user_id, NO session_id, NO raw
    # query text — only cause code, topic bucket, language, jurisdiction
    # (currently always null — no state field exists yet), and the best
    # retrieval candidate's score even when it fell below the confidence
    # threshold or nothing matched at all. This lets us tell "no law in the
    # corpus for this topic" apart from "a candidate existed but scored too
    # low" apart from "the query was too vague to score anything" — a
    # decision on where to spend corpus-building credits should never be
    # made without this breakdown.
    if early_refusal:
        if early_refusal == REFUSAL_NON_INDIAN:
            cause_code = "non_indian_jurisdiction"
            sub_cause = None
            dbg = {"top_score": None, "top_key": None, "content_tokens": None}
        elif early_refusal == REFUSAL_NOT_LEGAL:
            cause_code = "not_legal_advice_request"
            sub_cause = None
            dbg = {"top_score": None, "top_key": None, "content_tokens": None}
        elif personal_law_topic and personal_law_note is None:
            # This early_refusal is the disambiguation clarifying question,
            # not a genuine "nothing verified for this topic" refusal.
            cause_code = "personal_law_disambiguation"
            sub_cause = personal_law_topic
            dbg = {"top_score": None, "top_key": None, "content_tokens": None}
        else:
            cause_code = "no_corpus_match"
            dbg = top_candidate_debug(retrieval_text)
            if dbg["top_score"] == 0 and dbg["content_tokens"] <= 1:
                sub_cause = "query_too_vague"
            elif dbg["top_score"] == 0:
                sub_cause = "no_candidate_scored"  # true coverage gap OR a
                # phrasing mismatch against an entry that already exists —
                # only a human spot-check of a sample can tell those apart;
                # the score alone cannot.
            else:
                sub_cause = "below_confidence_threshold"  # candidate existed
                # (dbg["top_key"]) but scored under RETRIEVAL_MIN_SCORE — a
                # retrieval-tuning question, not necessarily a missing law.
        try:
            await db.refusal_events.insert_one({
                "id": str(uuid.uuid4()),
                "cause_code": cause_code,
                "sub_cause": sub_cause,
                "category": classify_topic(retrieval_text),
                "language": body.language,
                "state": user_state or None,
                "top_score": dbg["top_score"],
                "top_key": dbg["top_key"],
                "created_at": datetime.now(timezone.utc).isoformat(),
            })
        except Exception:
            logging.getLogger(__name__).warning("refusal_events insert failed", exc_info=True)

    # Build the verified-source context for the prompt (used ONLY when we have hits)
    corpus_context = ""
    if retrieved:
        corpus_context = "\n---\n".join([
            (
                f"STATE RULE — applies in {state_name(it['state'])} only: {it['scope_note']}"
                if it.get("state") else f"Scope: {it['scope_note']}"
            )
            for it in retrieved
        ])

    # Merge MongoDB corpus hits into context.  Safety guards (G1 dead-law, G2
    # judicial-invalidation) are encoded as SAFETY WARNING / JUDICIAL ALERT
    # lines BEFORE the statutory text so the LLM reads them first.
    if db_hits:
        db_parts: list[str] = []
        for hit in db_hits:
            lines: list[str] = []
            if hit.get("dead_warning"):
                lines.append(f"SAFETY WARNING (DEAD LAW): {hit['dead_warning']}")
            if hit.get("judicial_flag"):
                lines.append(f"JUDICIAL ALERT: {hit['judicial_flag']}")
            # Graduate Fix 1: surface registry_gap_note so the LLM is aware
            # the Act's current status has not been independently verified.
            if hit.get("registry_gap_note"):
                lines.append(f"REGISTRY NOTE: {hit['registry_gap_note']}")
            # P0 fix: the registry-level dead_warning above only names the
            # successor ACT (e.g. "Bharatiya Nyaya Sanhita, 2023"), never the
            # specific successor SECTION. When the seeded record itself
            # carries a section-specific successor_act (set at seed time —
            # see scripts/seed_ipc166a_bns199.py), surface that exact string
            # too so the LLM can actually name the successor section instead
            # of only the successor Act.
            #
            # CORRESPONDENCE GATE (Batch P2): a stored `successor_act` field
            # alone cannot authorize this claim.  It requires an active,
            # unrevoked section_correspondence approval in the registry.
            # Rechecked on every turn — never cached.
            if hit.get("successor_act"):
                _ipc_sec = (hit.get("section_number") or "").strip().upper()
                # Parse the BNS section number from the stored successor_act
                # string, e.g. "Bharatiya Nyaya Sanhita, 2023, Section 199" → "199"
                import re as _re_corr
                _bns_m = _re_corr.search(r"Section\s+(\S+)", hit.get("successor_act", ""), _re_corr.IGNORECASE)
                _bns_sec = _bns_m.group(1).rstrip(".,;") if _bns_m else ""
                _ipc_key = f"IPC:{_ipc_sec}" if _ipc_sec else ""
                _bns_key = f"BNS:{_bns_sec.upper()}" if _bns_sec else ""
                _corr_ok = (
                    _ipc_key and _bns_key and
                    _check_corr_approval(_ipc_key, _bns_key).approved
                )
                if _corr_ok:
                    lines.append(f"Successor provision: {hit['successor_act']}")
            sec_heading = hit.get("section_heading", "")
            act_label = f"{hit.get('act_name', '')}, Section {hit.get('section_number', '')}"
            if sec_heading:
                act_label += f" — {sec_heading}"
            lines.append(f"Source: {act_label}")
            if hit.get("no_current_text"):
                lines.append(
                    "NOTE: Current statutory text for this section is not available "
                    "in the corpus — the act may have been repealed or replaced by "
                    "successor legislation (e.g. IPC superseded by BNS). "
                    "Tell the user plainly that this section may no longer be in force "
                    "and advise them to consult an advocate for the current legal position."
                )
            else:
                sec_text = (hit.get("section_text") or "")[:600]
                if sec_text:
                    lines.append(f"Text: {sec_text}")
            db_parts.append("\n".join(lines))
        db_block = "\n---\n".join(db_parts)
        corpus_context = (corpus_context + "\n---\n" + db_block) if corpus_context else db_block

    # Orphan judicial warnings (section in judicial_invalidations but NOT in
    # legal_sections — e.g. IPC §377).  No statutory text available, but the
    # ruling itself is the verified source.
    if db_orphan:
        orphan_line = (
            f"JUDICIAL ALERT [{db_orphan.get('status', '')}]: {db_orphan.get('user_warning', '')}\n"
            f"Note: '{db_orphan.get('act_name', '')}' may no longer be in force in its original form "
            "(e.g. replaced by a successor code). Advise the user to consult an advocate for current law."
        )
        corpus_context = (corpus_context + "\n---\n" + orphan_line) if corpus_context else orphan_line
    # Tell the model, in plain words, when the local rule is the missing piece so
    # it never presents the central position as the complete answer.
    if state_topic and not state_hits:
        if user_state:
            corpus_context += (
                f"\n---\nJURISDICTION GAP: {state_topic['label']} are decided by STATE law and "
                f"no confirmed rule for {state_name(user_state)} is available. Say plainly that the "
                f"local rule decides this and that the user must confirm it with "
                f"{state_topic['authority']}. Do NOT state any local amount, limit or deadline."
            )
        else:
            corpus_context += (
                f"\n---\nJURISDICTION GAP: {state_topic['label']} are decided by STATE law and the "
                "user has not told us their state. Say plainly that the answer depends on their "
                "state and ask them to set their state in the app. Do NOT state any local amount, "
                "limit or deadline."
            )

    # Pro-quality prompt if user is Pro OR consuming a free sample; else basic prompt.
    use_pro_prompt = is_pro_user or is_sample_consumption

    # Daily LLM spend cap — only counted when we are actually going to call the
    # model (a refusal costs nothing, so it must not eat the user's allowance).
    if not early_refusal:
        await meter_llm_use(user, "question")

    # Derive native name — fall back to LANGUAGES table if client didn't send it
    lang_native = body.language_native
    if not lang_native:
        for _lang in LANGUAGES:
            if _lang["code"] == body.language:
                lang_native = _lang.get("native")
                break
    system_prompt = build_system_prompt(
        body.language_name,
        is_pro=use_pro_prompt,
        corpus_context=corpus_context,
        language_native=lang_native,
        advocate_mode=is_advocate_mode,
    )
    # Layer M: append status-guard reminder if any cited provision is dead.
    system_prompt = augment_prompt_with_status_guard(system_prompt, list(retrieved) + list(db_hits))
    # P0 fix: numeric fidelity for punishment/duration/fine figures — see
    # docstring. Applied whenever any verified source text was actually
    # assembled into corpus_context (i.e. there is something to be
    # unfaithful to); harmless no-op otherwise.
    if corpus_context:
        system_prompt = augment_prompt_with_numeric_fidelity_lock(system_prompt)
    # P0 fix: firmer instruction for the explicit-exact-dead-citation case
    # (see _has_explicit_dead_citation_with_successor above) — the generic
    # status-guard reminder alone was not enough to stop the LLM defaulting
    # to a refusal-style answer when a real citation was available.
    if _has_explicit_dead_citation_with_successor:
        system_prompt = augment_prompt_with_exact_dead_citation_lock(system_prompt)
    # ACTOR-CONDUCT LOCK: if Layer B found the user asking about someone
    # else's regulated conduct, instruct the LLM not to transfer that
    # obligation onto the user (see README_DEPLOYMENT.md > Engine Architecture).
    system_prompt = augment_prompt_with_actor_conduct_lock(system_prompt, _v2_classify)
    # DHARA-P0-4: immediate-danger guard (computed at the top of chat_stream).
    if _emergency:
        system_prompt += _EMERGENCY_PROMPT
        if not _DOMESTIC_FACT_RE.search(body.message or ""):
            retrieved = [it for it in retrieved if not _DV_ACT_RE.search(str(it.get("act", "")) + " " + str(it.get("citation", "")))]
            db_hits = [h for h in db_hits if not _DV_ACT_RE.search(str(h.get("act_name", "")))]

    chat = _ai_build_chat(session_id=session_id, system_message=system_prompt)

    def sse(obj: dict) -> bytes:
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8")

    async def save_assistant(content: str, error: Optional[str] = None):
        now_ts = datetime.now(timezone.utc).isoformat()
        # Infer status for CSV exports & analytics
        if error:
            status = "error"
        elif early_refusal == REFUSAL_NO_CORPUS:
            status = "refused_no_corpus"
        elif early_refusal == REFUSAL_NON_INDIAN:
            status = "refused_non_indian"
        elif early_refusal == REFUSAL_NOT_LEGAL:
            status = "refused_not_legal"
        elif personal_law_topic and personal_law_note is None and early_refusal:
            status = "personal_law_disambiguation_asked"
        else:
            status = "ok"
        doc = {
            "id": str(uuid.uuid4()),
            "session_id": session_id,
            "user_id": user["id"],
            "role": "assistant",
            "content": content,
            "language": body.language,
            "mode": mode,
            "model_provider": SERVER_CHAT_PROVIDER,
            "model_name": SERVER_CHAT_MODEL,
            "tier": "pro" if is_pro_user else "free",
            "sample_consumed": is_sample_consumption,
            "status": status,
            "created_at": now_ts,
            "timestamp": now_ts,
        }
        if error:
            doc["error"] = error
        # Also attach the citations that the user was shown, so the CSV export
        # can show what verified statutes accompanied each answer.
        if retrieved:
            doc["citations"] = [
                {"short_label": it.get("short_label", ""), "citation": it.get("citation", "")}
                for it in retrieved
            ]
        await db.messages.insert_one(doc)
        await db.sessions.update_one(
            {"id": session_id},
            {"$set": {"updated_at": datetime.now(timezone.utc).isoformat()}},
        )
        # Only debit a sample if the assistant actually returned real content (not empty error).
        if is_sample_consumption and content and not error:
            await db.users.update_one(
                {"id": user["id"]},
                {"$inc": {"pro_samples_used": 1}},
            )

    async def event_gen() -> AsyncGenerator[bytes, None]:
        yield sse({
            "type": "session",
            "session_id": session_id,
            "tier": "guest" if is_guest else ("pro" if is_pro_user else "free"),
            "mode": mode,
            "sample_consumed": is_sample_consumption,
            "samples_remaining_after": max(0, PRO_FREE_SAMPLES - samples_used - (1 if is_sample_consumption else 0)),
            "guest": is_guest,
            "guest_question_count": guest_question_count,
            "guest_question_limit": guest_question_limit,
            "guest_warning": guest_warning,
        })
        # Citation candidates are collected here first. With the flag OFF,
        # behaviour is byte-for-byte unchanged: they are flushed immediately
        # below. With CITATION_GUARD_ENABLED=true, they are held until the
        # answer itself is known to be substantive (not a refusal / not
        # blocked by the output guard) — see the post-processing block below.
        _citation_payloads: list[dict] = []
        for it in retrieved:
            _citation_payloads.append(
                {"type": "citation", "citation": with_historical_notice(public_citation(it), it)}
            )

        # Emit MongoDB-corpus citations.  G1/G2 safety warnings are prepended to
        # official_text so they appear at the top of the citation card in the UI.
        for hit in db_hits:
            dead_warn = hit.get("dead_warning") or ""
            ji_warn = hit.get("judicial_flag") or ""
            reg_note = hit.get("registry_gap_note") or ""
            warning_prefix = "\n\n".join(filter(None, [dead_warn, ji_warn, reg_note]))
            if hit.get("no_current_text"):
                no_text_notice = (
                    "⚠️ Current statutory text is not available for this section. "
                    "This may reflect a repeal or replacement by successor legislation. "
                    "Please consult an advocate for the current legal position."
                )
                official = (warning_prefix + "\n\n" + no_text_notice) if warning_prefix else no_text_notice
            else:
                raw_text = (hit.get("section_text") or "")[:2000]
                official = (warning_prefix + "\n\n" + raw_text) if warning_prefix else raw_text
            sec_heading = hit.get("section_heading", "")
            act_label = (
                f"{hit.get('act_name', '')}, Section {hit.get('section_number', '')}"
                + (f" — {sec_heading}" if sec_heading else "")
            )
            # CORRESPONDENCE GATE: only surface the specific successor section
            # if the IPC→BNS correspondence has an active, unrevoked approval.
            # Rechecked here on every turn — never cached.
            _raw_successor = hit.get("successor_act")
            _gated_successor_act = None
            if _raw_successor:
                import re as _re_cit
                _cit_ipc_sec = (hit.get("section_number") or "").strip().upper()
                _cit_bns_m   = _re_cit.search(r"Section\s+(\S+)", _raw_successor, _re_cit.IGNORECASE)
                _cit_bns_sec = _cit_bns_m.group(1).rstrip(".,;").upper() if _cit_bns_m else ""
                if _cit_ipc_sec and _cit_bns_sec:
                    _cit_corr_ok = _check_corr_approval(
                        f"IPC:{_cit_ipc_sec}", f"BNS:{_cit_bns_sec}"
                    ).approved
                    _gated_successor_act = _raw_successor if _cit_corr_ok else None

            _citation_payloads.append({
                "type": "citation",
                "citation": {
                    "key": f"{hit.get('act_name', '')} § {hit.get('section_number', '')}",
                    "citation": act_label,
                    "short_label": f"§ {hit.get('section_number', '')}",
                    "act": hit.get("act_name", ""),
                    "official_text": official,
                    "source_url": hit.get("source_url") or "",
                    "verified_at": hit.get("badge") or "Sourced from Government of India",
                    "text_kind": "verbatim" if not hit.get("no_current_text") else "judicial_ruling",
                    "state": None if (hit.get("jurisdiction") or "CENTRAL") == "CENTRAL" else hit.get("jurisdiction"),
                    "is_dead_law": hit.get("is_dead_law", False),
                    "dead_warning": hit.get("dead_warning"),
                    "successor_act": _gated_successor_act,
                    "judicial_flag": hit.get("judicial_flag"),
                    "no_current_text": hit.get("no_current_text", False),
                    # Graduate Fix 2: surface as a separate field so the
                    # frontend can render a distinct ℹ️ badge without parsing
                    # official_text.  Empty-string means "no gap note".
                    "registry_gap_note": reg_note,
                },
            })

        # Orphan judicial-invalidation warning (section in judicial_invalidations
        # but NOT in legal_sections — e.g. IPC §377 replaced by BNS).
        if db_orphan:
            _citation_payloads.append({
                "type": "citation",
                "citation": {
                    "key": f"{db_orphan.get('act_name', '')} § {db_orphan.get('section_number', '')}",
                    "citation": (
                        f"{db_orphan.get('act_name', '')}, Section {db_orphan.get('section_number', '')}"
                        f" [{db_orphan.get('status', '')}]"
                    ),
                    "short_label": f"§ {db_orphan.get('section_number', '')}",
                    "act": db_orphan.get("act_name", ""),
                    "official_text": db_orphan.get("user_warning", ""),
                    "source_url": "",
                    "verified_at": "Judicial Ruling",
                    "text_kind": "judicial_ruling",
                    "state": None,
                    "is_dead_law": False,
                    "judicial_flag": db_orphan.get("user_warning"),
                    "status": db_orphan.get("status"),
                },
            })

        if not CITATION_GUARD_ENABLED:
            # Unchanged legacy behaviour: emit every candidate immediately,
            # before the answer is even generated.
            for _p in _citation_payloads:
                yield sse(_p)

        # State-jurisdiction signalling. Two distinct cases, both honest:
        #  - the user has no state set → ask for it once, in context
        #  - the state is known but we hold no verified local rule yet → say so
        if state_topic and not state_hits:
            if not user_state:
                yield sse({
                    "type": "state_prompt",
                    "topic": state_topic["topic"],
                    "label": state_topic["label"],
                    "message": (
                        f"{state_topic['label'].capitalize()} are decided by state law. "
                        "Set your state once to get the rules that actually apply to you."
                    ),
                })
            else:
                yield sse({
                    "type": "state_note",
                    "topic": state_topic["topic"],
                    "state": user_state,
                    "message": (
                        f"No confirmed {state_topic['label']} rule for {state_name(user_state)} is "
                        f"in Dhara yet. Confirm the local position with {state_topic['authority']}."
                    ),
                })

        # DHARA C1-SOS: bare distress gate (fallback — fast-path above handles
        # most cases before run_pre_retrieval; this catches any edge-case that
        # reaches event_gen).  Runs BEFORE the deny-list check.
        if FEATURE_C1_CONVERSATIONAL_REPLIES and not _emergency and is_bare_distress(body.message):
            _sos_reply = get_conversational_template("distress_sos", body.language)
            if _sos_reply:
                yield sse({"type": "delta", "content": _sos_reply, "c1_sos": True})
                await save_assistant(_sos_reply, error=None)
                await save_case_state(db, session_id, _v2_state)
                yield sse({"type": "done"})
                return

        # DHARA C1 (GO amendment 1/2): conversational-intent gate. Runs
        # strictly AFTER the backend immediate-danger check above (_emergency)
        # and only when the message is not on the explicit deny list — a
        # denied or emergency message is NEVER given a canned reply and
        # always falls through to the normal path below unchanged.
        if (
            FEATURE_C1_CONVERSATIONAL_REPLIES
            and not _emergency
            and not is_denylisted(body.message)
        ):
            _c1_intent = classify_conversational_intent(body.message, body.language)
            if _c1_intent:
                _c1_reply = get_conversational_template(_c1_intent, body.language)
                if _c1_reply:
                    yield sse({"type": "delta", "content": _c1_reply})
                    await save_assistant(_c1_reply, error=None)
                    await save_case_state(db, session_id, _v2_state)
                    yield sse({"type": "done"})
                    return

        # DHARA Layer-S (section-level): IT Act §66A was struck down (Shreya
        # Singhal, 2015). Answer authoritatively instead of letting it fall
        # through to a generic "no active law" refusal. Runs after the
        # danger / SOS / C1 shortcuts and before the refusal / LLM path. No LLM
        # call, no fabricated replacement provision.
        if _is_it_66a_query(body.message):
            _66a_reply = _IT66A_NOTICE.get(body.language, _IT66A_NOTICE["en"])
            yield sse({"type": "delta", "content": _66a_reply, "law_status": "struck_down", "provision": "IT Act §66A"})
            await save_assistant(_66a_reply, error=None)
            await save_case_state(db, session_id, _v2_state)
            yield sse({"type": "done"})
            return

        # Refusal path — do not call the LLM. Send the refusal as a delta so the frontend
        # shows it in the normal chat bubble.
        if early_refusal:
            # Localize the refusal into the user's selected language (falls back to English)
            refusal_text = localize_refusal(early_refusal, body.language)
            yield sse({"type": "delta", "content": refusal_text})
            await save_assistant(refusal_text, error=None)
            await save_case_state(db, session_id, _v2_state)
            if CITATION_GUARD_ENABLED and _citation_payloads:
                logger.info("citation_guard | suppressed | reason=early_refusal | count=%d", len(_citation_payloads))
            if is_guest and guest_question_count == guest_question_limit:
                yield sse({"type": "guest_limit_reached", "message": "You've reached the guest limit. Create a free account to continue."})
            yield sse({"type": "done"})
            return

        full = ""
        errored: Optional[str] = None
        _buffering = CITATION_GUARD_ENABLED  # hold tokens, release only after the output guard passes
        if _buffering:
            yield sse({"type": "status", "state": "thinking"})
        try:
            async for ev in chat.stream_message(UserMessage(text=body.message)):
                if isinstance(ev, TextDelta):
                    full += ev.content
                    if not _buffering:
                        yield sse({"type": "delta", "content": ev.content})
                elif isinstance(ev, StreamDone):
                    break
        except Exception as e:
            logger.exception("LLM stream error")
            errored = str(e)[:300]
            yield sse({"type": "error", "error": errored})

        # Post-processing runs outside try/finally so that a `return` here can
        # never swallow an in-flight exception (the except above already handles
        # stream failures).
        # Citation-integrity: strip any leaked section/article/statute names from
        # the model's output. The verified citations are shown by the UI from the
        # `citation` frames — the model MUST NOT emit them itself.
        sanitized = sanitize_model_output(full) if full else full
        # ACTOR-CONDUCT LOCK: defence-in-depth — reframe any obligation
        # language still directly addressed to the user when Layer B found
        # an actor mismatch (prompt-level instruction is the primary guard).
        if sanitized:
            sanitized = reframe_misdirected_obligations(sanitized, _v2_classify)
        # P0 fix: deterministic correction for the recurring BNS 199 / IPC
        # 166A punishment-figure hallucination — see
        # correct_punishment_mismatch's docstring. No-op for every other
        # query (only matches when one of this turn's own hits is exactly
        # that section).
        if sanitized:
            sanitized = correct_punishment_mismatch(sanitized, _all_hits_for_status)

        # Reply-language enforcement. A user who selected Tamil and receives
        # English has been given nothing, so if the finished reply is not in the
        # expected script we translate it ONCE and overwrite the bubble.
        if sanitized and not errored and needs_language_repair(sanitized, body.language):
            lang_display = (
                f"{body.language_name} ({lang_native})"
                if lang_native and lang_native != body.language_name
                else body.language_name
            )
            logger.warning("reply-language repair: model answered outside the %s script", lang_display)
            try:
                fixer = _ai_build_chat(
                    session_id=f"{session_id}-langfix",
                    system_message=repair_prompt(lang_display),
                )
                translated = await fixer.send_message(UserMessage(text=sanitized))
                translated = sanitize_model_output(str(translated or "")).strip()
                if translated and not needs_language_repair(translated, body.language):
                    sanitized = translated
            except Exception:
                logger.exception("reply-language repair failed")

        # Append the personal-law disclaimer (see above) so the user always
        # sees which religion/marriage-type's law was applied.
        if personal_law_note and not errored and (sanitized or full):
            sanitized = (sanitized or full) + f"\n\n{personal_law_note}"

        # DHARA-P0-4: always lead an immediate-danger answer with 112.
        if _emergency and not errored and (sanitized or full):
            _banner = _EMERGENCY_BANNER.get(body.language, _EMERGENCY_BANNER["en"])
            sanitized = f"{_banner}\n\n{sanitized or full}"

        # ── Phase 1 hotfix: final output guard + citation-after-answer gate ──
        # Runs ONLY on this DHARA-authored prose variable — never on a
        # citation's official_text (those are separate payloads, never
        # passed through validate_dhara_text). Logs reason codes/counts
        # only — never the question, answer or citation text itself.
        _suppress_citations = False
        if CITATION_GUARD_ENABLED and not errored:
            _candidate = sanitized or full
            _clean, _blocked, _repeats = validate_dhara_text(_candidate)
            if _blocked:
                logger.warning("citation_guard | output_blocked | reason=unverified_marker_or_banned_word")
                sanitized = CANNOT_CONFIRM_MESSAGE
                _suppress_citations = True
            else:
                sanitized = _clean
                if _repeats:
                    logger.info("citation_guard | repeats_collapsed | count=%d", _repeats)
                if looks_like_refusal(sanitized):
                    # P0 fix (BUG 2/3 follow-up): the same successor-aware
                    # exemption used at the Layer M gate above must also
                    # apply here. Without it, this second guard independently
                    # re-detected the LLM's own refusal-flavoured prose (e.g.
                    # "I cannot confirm this is in force, consult an
                    # advocate") for an EXPLICIT exact-section dead-law
                    # lookup that DOES have a verified citation + successor —
                    # and threw that real citation away anyway, reproducing
                    # the exact bug the Layer M fix was meant to close.
                    if _has_explicit_dead_citation_with_successor:
                        logger.info(
                            "citation_guard | refusal_text_detected_but_kept | "
                            "reason=explicit_exact_dead_citation_with_successor | count=%d",
                            len(_citation_payloads),
                        )
                    else:
                        _suppress_citations = True
                        logger.info("citation_guard | suppressed | reason=refusal_text_detected | count=%d", len(_citation_payloads))

        if _buffering or sanitized != full:
            # Overwrite the accumulated text on the client with the sanitized version.
            # (Buffering mode never sent any delta, so this is the client's first
            # and only content for this turn — same "final" event type as today.)
            yield sse({"type": "final", "content": sanitized})
        if CITATION_GUARD_ENABLED and not _suppress_citations:
            for _p in _citation_payloads:
                yield sse(_p)
        await save_assistant(sanitized or full, errored)
        # Gate 2: persist updated conversation state (turn count, legal_query, domain)
        await save_case_state(db, session_id, _v2_state)
        if is_guest and guest_question_count == guest_question_limit:
            yield sse({"type": "guest_limit_reached", "message": "You've reached the guest limit. Create a free account to continue."})
        yield sse({"type": "done"})

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )

@router.get("/chat/sessions")
async def list_sessions(user: dict = Depends(current_user)):
    sessions = await db.sessions.find({"user_id": user["id"]}, {"_id": 0}).sort("updated_at", -1).to_list(200)
    return sessions

@router.get("/chat/sessions/{session_id}/messages")
async def session_messages(session_id: str, user: dict = Depends(current_user)):
    session = await db.sessions.find_one({"id": session_id, "user_id": user["id"]}, {"_id": 0})
    if not session:
        raise HTTPException(404, "Session not found")
    msgs = await db.messages.find({"session_id": session_id, "user_id": user["id"], "purged": {"$ne": True}}, {"_id": 0}).sort("created_at", 1).to_list(1000)
    return {"session": session, "messages": msgs}

@router.delete("/chat/sessions/{session_id}")
async def delete_session(session_id: str, user: dict = Depends(current_user)):
    await db.sessions.delete_one({"id": session_id, "user_id": user["id"]})
    await db.messages.delete_many({"session_id": session_id, "user_id": user["id"]})
    return {"ok": True}

# ---------- Saved answers (bookmarks) ----------
# The device keeps its own copy in AsyncStorage so a saved answer opens with no
# network at all (the "standing in a police station" case). The server copy is
# only a sync target so the collection survives a reinstall or a new phone.


@router.get("/bookmarks")
async def list_bookmarks(user: dict = Depends(current_user)):
    items = await db.bookmarks.find(
        {"user_id": user["id"], "deleted": {"$ne": True}}, {"_id": 0, "user_id": 0}
    ).sort("created_at", -1).to_list(500)
    return items


@router.put("/bookmarks")
async def upsert_bookmark(body: BookmarkIn, user: dict = Depends(current_user)):
    """Idempotent on (user, client_id) so the device can re-push its queue after
    being offline without creating duplicates."""
    now = datetime.now(timezone.utc).isoformat()
    doc = {
        "client_id": body.client_id,
        "user_id": user["id"],
        "question": body.question,
        "answer": body.answer,
        "language": body.language,
        "citations": body.citations,
        "created_at": body.created_at or now,
        "deleted": False,
        "synced_at": now,
    }
    await db.bookmarks.update_one(
        {"user_id": user["id"], "client_id": body.client_id},
        {"$set": doc, "$setOnInsert": {"id": str(uuid.uuid4())}},
        upsert=True,
    )
    saved = await db.bookmarks.find_one(
        {"user_id": user["id"], "client_id": body.client_id}, {"_id": 0, "user_id": 0}
    )
    return saved


@router.delete("/bookmarks/{client_id}")
async def delete_bookmark(client_id: str, user: dict = Depends(current_user)):
    # Soft delete — a tombstone keeps a second device from resurrecting the row.
    await db.bookmarks.update_one(
        {"user_id": user["id"], "client_id": client_id},
        {"$set": {"deleted": True, "deleted_at": datetime.now(timezone.utc).isoformat()}},
    )
    return {"ok": True}

# ---------- Ready-to-send notice drafts ----------
# The draft text itself is assembled ON THE DEVICE from a fixed template (no LLM
# call, so it is deterministic and costs nothing). This endpoint only meters the
# entitlement: the first draft is free, after that it is a Pro feature.
DRAFT_TYPES = {"cheque_bounce", "deposit_refund", "unpaid_salary"}


@router.post("/drafts/consume")
async def consume_draft(payload: dict, user: dict = Depends(current_user)):
    draft_type = (payload.get("draft_type") or "").strip()
    if draft_type not in DRAFT_TYPES:
        raise HTTPException(400, "Unknown draft type")
    used = int(user.get("drafts_used", 0))
    if not user.get("is_pro") and used >= DRAFTS_FREE:
        raise HTTPException(
            status_code=402,
            detail={
                "paywall": True,
                "reason": "drafts_exhausted",
                "drafts_used": used,
                "drafts_free_limit": DRAFTS_FREE,
                "pro_price_label": PRO_PRICE_LABEL,
                "pro_price_usd_label": PRO_PRICE_USD_LABEL,
                "message": (
                    f"Your {DRAFTS_FREE} free notice draft has been used. Upgrade to Pro for "
                    "unlimited ready-to-send legal notices."
                ),
            },
        )
    if not user.get("is_pro"):
        await db.users.update_one({"id": user["id"]}, {"$inc": {"drafts_used": 1}})
        used += 1
    await db.draft_events.insert_one({
        "id": str(uuid.uuid4()),
        "user_id": user["id"],
        "draft_type": draft_type,
        "is_pro": bool(user.get("is_pro")),
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    return {
        "ok": True,
        "draft_type": draft_type,
        "drafts_used": used,
        "drafts_remaining": max(0, DRAFTS_FREE - used),
    }

# ---------- Voice ----------

def _script_check(text: str, language: str | None) -> dict:
    """check_script_mismatch() plus the `script_mismatch` key the app reads."""
    result = check_script_mismatch(text, language)
    return {**result, "script_mismatch": result["mismatch"]}


MAX_AUDIO_BYTES = 10 * 1024 * 1024   # Whisper hard limit is 25 MB; ≈10 min of AAC voice
MIN_AUDIO_BYTES = 1024
_GENERIC_AUDIO_TYPES = {"", "application/octet-stream"}
_DECLARED_FAMILY = {
    "audio/m4a": "mp4", "audio/x-m4a": "mp4", "audio/mp4": "mp4", "video/mp4": "mp4", "audio/aac": "mp4",
    "audio/webm": "webm", "video/webm": "webm",
    "audio/ogg": "ogg", "audio/opus": "ogg",
    "audio/mpeg": "mp3", "audio/mp3": "mp3",
    "audio/wav": "wav", "audio/x-wav": "wav", "audio/wave": "wav", "audio/vnd.wave": "wav",
    "audio/flac": "flac", "audio/x-flac": "flac",
}
_FAMILY_FILE = {"mp4": ("audio.m4a", "audio/m4a"), "webm": ("audio.webm", "audio/webm"),
                "ogg": ("audio.ogg", "audio/ogg"), "mp3": ("audio.mp3", "audio/mpeg"),
                "wav": ("audio.wav", "audio/wav"), "flac": ("audio.flac", "audio/flac")}


def sniff_audio_family(data: bytes) -> Optional[str]:
    """Identify the container from its magic bytes (never trust the name)."""
    if len(data) >= 8 and data[4:8] == b"ftyp":
        return "mp4"
    if data[:4] == b"\x1a\x45\xdf\xa3":
        return "webm"
    if data[:4] == b"OggS":
        return "ogg"
    if data[:4] == b"fLaC":
        return "flac"
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "wav"
    if data[:3] == b"ID3" or (len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0):
        return "mp3"
    return None


async def read_validated_audio(audio: UploadFile) -> tuple[bytes, str, str]:
    """Size / emptiness / declared-type / signature checks for every voice
    upload. Runs BEFORE metering so rejected files are never billed.
    Returns (bytes, filename, content_type) normalised to the sniffed format."""
    data = await audio.read(MAX_AUDIO_BYTES + 1)
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(413, "Recording is too long (max 10 MB). Please record a shorter message.")
    if len(data) < MIN_AUDIO_BYTES:
        raise HTTPException(400, "Recording was empty or too short. Please hold the mic button and speak for at least a second.")
    declared = (audio.content_type or "").split(";")[0].strip().lower()
    if declared not in _GENERIC_AUDIO_TYPES and declared not in _DECLARED_FAMILY:
        raise HTTPException(415, "This audio format isn't supported.")
    family = sniff_audio_family(data)
    if family is None or (declared in _DECLARED_FAMILY and _DECLARED_FAMILY[declared] != family):
        raise HTTPException(415, "This audio format isn't supported.")
    filename, ctype = _FAMILY_FILE[family]
    return data, filename, ctype


async def whisper_transcribe(audio: tuple[bytes, str, str], language: Optional[str]) -> dict:
    """FastAPI-facing wrapper; delegates core logic to ``stt_engine``.
    ``audio`` must come from ``read_validated_audio``."""
    from engine.stt_engine import transcribe_with_retry
    data, filename, ctype = audio
    try:
        oc = openai_client()
        return await transcribe_with_retry(
            audio_bytes=data,
            filename=filename,
            content_type=ctype,
            language=language,
            openai_client=oc,
            script_check_fn=_script_check,
        )
    except Exception as e:
        err_msg = str(e).lower()
        if "audio_too_short" in err_msg or "too short" in err_msg:
            logger.info("Whisper rejected clip as too short")
            raise HTTPException(
                400,
                "Recording was too short. Please hold the mic button and speak for at least a second.",
            )
        logger.exception("transcribe failed")
        raise HTTPException(500, f"Transcription failed: {e}")


@router.post("/voice/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    language: str = Form(None),
    user: dict = Depends(current_user),
):
    clip = await read_validated_audio(audio)   # validate first — rejected files are never billed
    await meter_llm_use(user, "voice")
    return await whisper_transcribe(clip, language)

@router.post("/voice/tts")
async def tts(body: TTSIn, user: dict = Depends(current_user)):
    # Reject empty / whitespace-only text with a clean 400 instead of letting
    # OpenAI's own 400 bubble up as a generic 500 — the frontend caches the
    # response as an MP3 and would otherwise write the JSON error blob to
    # disk, causing createAudioPlayer to fail silently.
    if not body.text or not body.text.strip():
        raise HTTPException(400, "Text is required to synthesise speech.")

    # The client splits an answer into sentences and requests them separately, so
    # the first words start playing in ~2s instead of after the whole answer has
    # been synthesised (which measured 13s for a 990-char reply). Metering every
    # request would then charge a free user ~6 of their 30 daily questions for a
    # single answer, so an answer costs one: only the first chunk of a `group`
    # meters, the rest ride along.
    #
    # Reuses usage_daily deliberately — it is already partitioned by day like the
    # counters, so this needs no new collection and no TTL index (there is no
    # index-creation or startup hook in this service to hang one off).
    should_meter = True
    if body.group:
        _day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        _res = await db.usage_daily.update_one(
            {
                "scope": "user",
                "user_id": user["id"],
                "day": _day,
                "kind": f"voice_group:{body.group[:64]}",
            },
            {"$setOnInsert": {"count": 1}},
            upsert=True,
        )
        # Inserted => first chunk of this answer => charge. Matched => already paid.
        should_meter = _res.upserted_id is not None
    if should_meter:
        await meter_llm_use(user, "voice")
    try:
        oc = openai_client()
        resp = await oc.audio.speech.create(
            model="tts-1",
            voice=body.voice,
            input=body.text[:4000],
            response_format="mp3",
        )
        audio_bytes = await resp.aread()
        return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/mpeg")
    except Exception as e:
        logger.exception("tts failed")
        raise HTTPException(500, f"TTS failed: {e}")

@router.post("/chat/followup")
async def chat_followup(body: ChatFollowupIn, user: dict = Depends(current_user)):
    """Generate 3 short follow-up question chips after an AI chat answer."""
    if not body.answer or len(body.answer.strip()) < 20:
        return {"questions": []}

    # Skip for C1 / distress turns — greetings, thanks and SOS replies have
    # no legal topic to follow up on, and generating chips wastes an LLM call.
    if FEATURE_C1_CONVERSATIONAL_REPLIES and (
        is_bare_distress(body.message)
        or (
            not is_denylisted(body.message)
            and bool(classify_conversational_intent(body.message, body.language))
        )
    ):
        return {"questions": []}

    lang_instruction = f"in {body.language_name}" if body.language_name.lower() != "english" else "in English"
    prompt = f"""A citizen consulted an Indian legal assistant about a specific legal matter.

Original question: "{body.message[:200]}"
Assistant's answer (first 400 chars): "{body.answer[:400]}"

Generate exactly 3 SHORT follow-up question chips {lang_instruction} that represent the citizen's NEXT LOGICAL STEPS for this specific legal topic.

Strict rules:
- Each chip must be under 9 words
- All 3 chips must be directly answerable from Indian law, statutes, or official procedures — never meta-questions about app features or capabilities
- Focus on: time limits / deadlines mentioned, the filing or escalation process, penalties or enforcement, related rights or remedies
- Chip 1: the most immediate next practical step (e.g. "What is the 30-day notice period?")
- Chip 2: the process / how to take action (e.g. "Can I file the FIR online?")  
- Chip 3: consequences or enforcement (e.g. "What court handles cheque-bounce cases?")
- Phrased as a brief question — not statements
- No duplicate of the original question

Reply ONLY with a valid JSON array of 3 strings, nothing else:
["chip 1", "chip 2", "chip 3"]"""

    try:
        import json as _json
        chat_llm = _ai_build_chat(
            session_id=f"chat-followup-{id(body)}",
            system_message="Generate exactly 3 short follow-up legal question chips as a JSON array only. All chips must be answerable from Indian law.",
        )

        result = await chat_llm.send_message(UserMessage(text=prompt))
        raw = (result or "[]").strip()
        start = raw.find('[')
        end = raw.rfind(']') + 1
        if start >= 0 and end > start:
            questions = _json.loads(raw[start:end])
            if isinstance(questions, list):
                valid = [str(q).strip() for q in questions if isinstance(q, str) and q.strip()][:3]
                return {"questions": valid}
    except Exception as e:
        logger.error(f"[chat_followup] Error: {e}")

    return {"questions": []}
