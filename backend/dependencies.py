"""Shared database connection, authentication utilities, and LLM helpers.

Imported by all API routers. Creates the single MongoDB connection used
across the entire backend.
"""
import bcrypt
import hmac
import jwt
import os
import uuid
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import HTTPException, Header
from motor.motor_asyncio import AsyncIOMotorClient
from services.ai_gateway import send_once, RETRIEVAL_KEYWORD_SPEC

from config.settings import (
    MONGO_URL, DB_NAME, CORPUS_DB_NAME,
    JWT_SECRET, JWT_ALG, JWT_EXP_DAYS,
    PRO_FREE_SAMPLES, DRAFTS_FREE,
    FREE_DAILY_QUERIES, APP_DAILY_LLM_CALLS,
    EMERGENT_LLM_KEY,
)
from corpus_db import STATE_CODE_TO_JURISDICTION
from states import state_name
from corpus import REFUSAL_NO_CORPUS, REFUSAL_NOT_LEGAL

logger = logging.getLogger("gandhikar")

# ── Single shared MongoDB connection ──────────────────────────────────────────
_mongo_client = AsyncIOMotorClient(MONGO_URL)
db             = _mongo_client[DB_NAME]
corpus_db      = _mongo_client[CORPUS_DB_NAME]

# ── Auth helpers ──────────────────────────────────────────────────────────────

def hash_pw(pw: str) -> str:
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()


def check_pw(pw: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(pw.encode(), hashed.encode())
    except Exception:
        return False


# ── Admin auth ──────────────────────────────────────────────────────────────
# Shared by all admin-only endpoints (CSV export in server.py, journeys admin
# import in api/journeys_router.py, etc.) — moved here from server.py so
# non-server modules can use it too without a circular import. The key MUST
# be sent in the X-Admin-Key header (never a ?key= query param, which leaks
# into access logs / proxy logs / browser history):
#   curl -H "X-Admin-Key: $ADMIN_KEY" <API>/api/admin/...
ADMIN_KEY = os.environ.get("ADMIN_KEY", "").strip()


def check_admin_key(key: Optional[str]) -> None:
    if not ADMIN_KEY:
        raise HTTPException(503, "Admin access is not configured on this server.")
    if not key or not hmac.compare_digest(key, ADMIN_KEY):
        raise HTTPException(401, "Invalid admin key.")


def make_token(user_id: str) -> str:
    payload = {
        "sub": user_id,
        "exp": datetime.now(timezone.utc) + timedelta(days=JWT_EXP_DAYS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)


def public_user(u: dict) -> dict:
    state_code = u.get("state") or ""
    return {
        "id": u["id"],
        "email": u["email"],
        "name": u["name"],
        "phone": u.get("phone", ""),
        "language": u.get("language", "en"),
        "state": state_code or None,
        "state_name": state_name(state_code),
        "is_pro": bool(u.get("is_pro", False)),
        "pro_since": u.get("pro_since"),
        "pro_samples_used": int(u.get("pro_samples_used", 0)),
        "pro_samples_limit": PRO_FREE_SAMPLES,
        "pro_samples_remaining": max(0, PRO_FREE_SAMPLES - int(u.get("pro_samples_used", 0))),
        "drafts_used": int(u.get("drafts_used", 0)),
        "drafts_free_limit": DRAFTS_FREE,
        "drafts_remaining": max(0, DRAFTS_FREE - int(u.get("drafts_used", 0))),
        "is_grandfathered": bool(u.get("is_grandfathered", True)),
        "state_jurisdiction": (
            u.get("state_jurisdiction")
            or STATE_CODE_TO_JURISDICTION.get(state_code.upper(), "")
        ),
        "terms_accepted": u.get("terms_accepted", False),
        "terms_version": u.get("terms_version"),
        "terms_accepted_at": u.get("terms_accepted_at"),
        # Profile photo (P1) — additive; never embeds image bytes in the user doc.
        "photo_url": (f"/api/profile/photo/{u['id']}?v={u.get('photo_version')}" if u.get("photo_path") else None),
        "photo_visibility": u.get("photo_visibility", "only_me"),
    }


async def current_user(authorization: Optional[str] = Header(None)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing token")
    token = authorization.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
    except jwt.PyJWTError:
        raise HTTPException(401, "Invalid token")
    user = await db.users.find_one({"id": payload["sub"]}, {"_id": 0})
    if not user:
        raise HTTPException(401, "User not found")
    if user.get("age_confirmed_18") is False:
        raise HTTPException(403, {
            "code": "minor_blocked",
            "message": (
                "Please ask a parent or guardian to help you set up Dhara. "
                "You can still use Emergency Helplines and Know Your Rights "
                "without an account."
            ),
            "helplines_url": "/api/emergency/helplines",
            "rights_url":    "/api/legal/categories",
        })
    return user


async def current_user_optional(authorization: Optional[str] = Header(None)) -> Optional[dict]:
    """Returns None instead of raising 401 when unauthenticated."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
    except jwt.PyJWTError:
        return None
    return await db.users.find_one({"id": payload["sub"]}, {"_id": 0})


# ── LLM spend metering ────────────────────────────────────────────────────────

async def meter_llm_use(user: dict, kind: str) -> None:
    """Count one paid LLM call. Raises 429 when caps are exceeded."""
    day    = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    is_pro = bool(user.get("is_pro"))

    app_doc = await db.usage_daily.find_one_and_update(
        {"scope": "app", "day": day},
        {"$inc": {"count": 1}},
        upsert=True,
        return_document=True,
    )
    if int(app_doc.get("count", 0)) > APP_DAILY_LLM_CALLS:
        raise HTTPException(
            status_code=429,
            detail={
                "limit": True,
                "scope": "app",
                "message": "Dhara is unusually busy today and has paused new answers. Please try again tomorrow.",
            },
        )

    if is_pro:
        return

    cap = FREE_DAILY_QUERIES
    doc = await db.usage_daily.find_one_and_update(
        {"scope": "user", "user_id": user["id"], "day": day, "kind": "query"},
        {"$inc": {"count": 1}},
        upsert=True,
        return_document=True,
    )
    used = int(doc.get("count", 0))
    if used > cap:
        raise HTTPException(
            status_code=429,
            detail={
                "limit": True,
                "scope": "user",
                "kind": "query",
                "used": used - 1,
                "cap": cap,
                "is_pro": False,
                "message": (
                    f"You've used all {cap} free questions for today. "
                    "Upgrade to Pro for unlimited access, or come back tomorrow.\n\n"
                    "In an emergency, free help is available:\n"
                    "• NALSA Legal Aid: 15100 (free)\n"
                    "• Consumer Helpline: 1800-11-4000 (toll-free)"
                ),
                "helplines": [
                    {"name": "NALSA Legal Aid", "number": "15100"},
                    {"name": "Consumer Helpline", "number": "1800-11-4000"},
                ],
            },
        )


# ── Retrieval translation (non-English → English keywords for corpus) ─────────

async def translate_for_retrieval(text: str) -> str:
    """Translate a non-English query to English legal keywords for corpus lookup.

    Returns "" on failure so the caller sees an honest REFUSAL_NO_CORPUS
    instead of spurious matches from raw Devanagari tokens.
    """
    import asyncio as _aio

    def _is_mostly_english(s: str) -> bool:
        if not s:
            return False
        non_ascii = sum(1 for c in s if ord(c) >= 128)
        return non_ascii / max(len(s), 1) < 0.25

    _TRANSLATE_SYSTEM = (
        "You are a legal-domain keyword extractor for an Indian law retrieval engine. "
        "Given a question in ANY Indian language, output 5-8 space-separated English keywords "
        "that identify the LEGAL DOMAIN and SPECIFIC ACT — NOT a literal word-for-word translation.\n\n"
        "CRITICAL RULES:\n"
        "1. First 1-2 keywords MUST name the legal domain: religious | criminal | property | "
        "family | employment | contract | traffic | data | civil\n"
        "2. Avoid generic action verbs (enter, go, come, visit, give, take) — they cause wrong matches.\n"
        "3. Use ACT NAMES when the query implies one (BNS, BNSS, Constitution, Hindu Marriage Act, etc.).\n"
        "4. NEVER output a sentence. NEVER output punctuation. Output keywords only.\n\n"
        "EXAMPLES (learn the pattern):\n"
        "'मस्जिद में हिंदू जाए तो क्या होगा?' → religious Hindu Muslim place worship visitor rights trespass\n"
        "'मुझे पुलिस ने बिना वारंट के पकड़ा' → criminal arrest warrant police custody rights Constitution BNS\n"
        "'दहेज के लिए उत्पीड़न क्या कानून है?' → family dowry harassment cruelty BNS Dowry Prohibition Act\n"
        "'किरायेदार को बेदखल कर सकते हैं?' → property tenant eviction rent landlord Transfer Property Act\n"
    )

    async def _attempt(attempt_num: int):
        try:
            result = await send_once(
                f"rtranslate-{uuid.uuid4()}",
                _TRANSLATE_SYSTEM,
                text,
                spec=RETRIEVAL_KEYWORD_SPEC,
                caller="retrieval_translate",
                timeout_s=RETRIEVAL_KEYWORD_SPEC.timeout_s,
            )
            out = str(result or "").strip().strip('"').strip()
            if out and _is_mostly_english(out) and out != text:
                logger.info("rtranslate | ok | attempt=%d | q=%r → %r", attempt_num, text[:60], out[:60])
                return out
            logger.warning("rtranslate | bad_output | attempt=%d | got=%r", attempt_num, out[:60])
            return None
        except _aio.TimeoutError:
            logger.warning("rtranslate | timeout(12s) | attempt=%d | q=%r", attempt_num, text[:60])
            return None
        except Exception as exc:
            logger.warning("rtranslate | exception | attempt=%d | %s", attempt_num, exc, exc_info=False)
            return None

    result = await _attempt(1)
    if result:
        return result
    result = await _attempt(2)
    if result:
        return result

    logger.warning("rtranslate | GIVING_UP | returning empty → REFUSAL_NO_CORPUS | q=%r", text[:60])
    return ""


# ── System prompt builder ─────────────────────────────────────────────────────

def build_system_prompt(
    language_name: str,
    is_pro: bool = False,
    corpus_context: str = "",
    language_native: Optional[str] = None,
    advocate_mode: bool = False,
) -> str:
    """Grade-6 answer prompt with strict citation-integrity rule."""
    lang_display = (
        f"{language_name} ({language_native})"
        if language_native and language_native != language_name
        else language_name
    )
    is_english = language_name.strip().lower() == "english"

    if advocate_mode:
        verified_block_adv = ""
        if corpus_context:
            verified_block_adv = (
                "\n\nSOURCES (cite these directly by Act name and section):\n"
                + corpus_context + "\n"
            )
        return (
            "You are Dhara — an AI legal analysis assistant for licensed Indian advocates.\n\n"
            "PROFESSIONAL MODE: You are addressing a licensed Indian advocate. "
            "Use formal legal language. You MAY cite specific Act names, section numbers, "
            "sub-sections, and provisos found in the SOURCES below. "
            "Assume the reader has legal training. "
            "Flag areas where the advocate should confirm with current gazette notifications or state amendments.\n\n"
            "HARD RULES:\n"
            "1. Base your analysis ONLY on the SOURCES provided. "
            "If SOURCES are empty, decline with: "
            "'Insufficient corpus material for a professional brief on this topic. Please consult primary sources.'\n"
            "2. No markdown headers (#, ##). No bold (**). No emoji.\n"
            "3. If the corpus does not cover this topic, state so explicitly rather than extrapolating.\n\n"
            "FORMAT — respond in this four-part structure:\n\n"
            "Issue: <one-sentence statement of the legal question>\n\n"
            "Applicable Law: <cite the Act(s) and section(s) from the sources>\n\n"
            "Legal Position: <technical analysis in 3–5 sentences>\n\n"
            "Practitioner Notes:\n"
            "- <procedural step or confirmation point>\n"
            "- <state amendment flag or gazette check needed>\n"
            "- <escalation path or time-limit note>\n"
            "- <optional 4th note>"
            f"{verified_block_adv}"
        )

    disclaimer_line = (
        "\n\nAt the end of your reply, add exactly this line, WRITTEN IN " + lang_display + ":\n"
        '"⚠️ Legal information, not legal advice. Consult an advocate. © Callistus Moses · MSafe Solutions."'
    )
    verified_block = ""
    if corpus_context:
        verified_block = (
            "\n\nSOURCES (already shown to the user by the UI — DO NOT repeat verbatim, "
            "DO NOT quote, DO NOT emit section/article numbers). Use these to shape your plain-language "
            "explanation ONLY:\n" + corpus_context + "\n"
        )

    language_rule = (
        f"REPLY LANGUAGE: {lang_display}. Write your ENTIRE reply — every word, every heading, "
        f"every bullet, and the final disclaimer — in {lang_display}. Use the {lang_display} script. "
        "The ONLY things that may stay in English are widely-used acronyms like FIR, RTI, POSH, PIO, SP, "
        "and proper nouns. Do NOT reply in English if the language above is not English. "
        "If you catch yourself writing in English, translate everything before answering."
        if not is_english else
        "REPLY LANGUAGE: English. Simple, dignified, Grade 6 reading level. Talk to the person as \"you\"."
    )
    label_hint = (
        ""
        if is_english else
        f"\n\nIMPORTANT — TRANSLATE THE SECTION HEADINGS. In your reply, the two section headings "
        f"(the equivalents of 'Answer:' and 'What you can do:') MUST be written in {lang_display}, "
        f"NOT in English. The user is a native {lang_display} speaker and will not understand English headings."
    )

    base_rules = f"""You are Dhara — an AI legal information assistant for Indian citizens.

{language_rule}{label_hint}

HARD RULES (do not break — the app will strip your reply if you break them):
1. NEVER write section numbers, article numbers, or clause numbers in your reply. Do NOT write "Article 21", "Section 35", "BNS", "BNSS", "BNSS 43(5)", "Section", "Article" etc. anywhere in your reply text. The user sees the exact citations in a separate box below your reply.
2. NEVER quote statutory text verbatim. Do NOT copy the words of the law. Only explain in your own plain words.
3. NEVER use markdown headers (# or ##). No bold with **. No emoji.
4. NEVER use Latin phrases or unexplained legal jargon.
5. If the law has an exception (except, unless, provided that, save in), your reply MUST mention the exception plainly. Never state a right as absolute if the law qualifies it.
6. If your SOURCES block is empty or missing, reply exactly: "{REFUSAL_NO_CORPUS}" and STOP.
7. If the question is about non-Indian law, or asks for personal/moral advice (should I forgive, should I marry, etc.), reply exactly: "{REFUSAL_NOT_LEGAL}" and STOP.

FORMAT — respond in exactly this two-part shape and nothing else. The two headings below appear here in English as placeholders; you MUST write them in {lang_display} in your reply:

<heading meaning "Answer:" in {lang_display}> <the direct answer in AT MOST TWO sentences, each under 15 words. First sentence must be the answer.>

<heading meaning "What you can do:" in {lang_display}>
- <one short bullet, plain action in {lang_display}>
- <one short bullet, plain action in {lang_display}>
- <one short bullet, plain action in {lang_display}>
- <optional 4th bullet in {lang_display}>
- <optional 5th bullet in {lang_display}>

Then a blank line, then the disclaimer line (also in {lang_display}).{verified_block}{disclaimer_line}
"""
    if is_pro:
        return base_rules + (
            f"\n\nPRO MODE: after the standard reply above, add another section titled with the {lang_display} "
            f"equivalent of 'For your situation:' (in {lang_display}) with 3–4 more bullets giving a step-by-step "
            f"action plan (offices to visit, forms, escalation contacts) — all in {lang_display}. "
            "Same rules — no section numbers, no verbatim law, no markdown headers."
        )
    return base_rules
