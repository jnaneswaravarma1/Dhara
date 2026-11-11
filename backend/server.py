"""Dhara - AI Legal Empowerment Bot Backend."""
import os
import io
import re
import json
import httpx
import uuid
import asyncio
import hmac
import hashlib
import secrets
import logging
import bcrypt
import jwt
import stripe
try:
    import razorpay
except Exception:
    razorpay = None
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from pathlib import Path
from typing import List, Optional, AsyncGenerator

from fastapi import FastAPI, APIRouter, HTTPException, Depends, UploadFile, File, Form, Header, Request, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from pydantic import BaseModel, EmailStr, Field
from openai import AsyncOpenAI

from emergentintegrations.llm.chat import TextDelta, StreamDone
from services.ai_gateway import send_once as _gateway_send_once, INTAKE_SUMMARY_SPEC

from legal import TERMS_AND_CONDITIONS, TERMS_VERSION, DISCLAIMER_SHORT
from mailer import send_email, password_reset_otp_email, account_deletion_otp_email, email_configured
from corpus import (
    retrieve as corpus_retrieve,
    retrieve_state as corpus_retrieve_state,
    state_sensitive_topic,
    is_non_indian_jurisdiction,
    is_non_legal_advice,
    public_citation,
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
    STATE_CODE_TO_JURISDICTION,
)
from states import STATES, STATE_BY_CODE, is_valid_state, state_name
from langpolicy import needs_language_repair, repair_prompt, needs_retrieval_translation
from fir_storage import init_storage as _init_fir_storage
from config.settings import (
    SERVER_CHAT_MODEL, SERVER_CHAT_PROVIDER, EMERGENT_LLM_KEY,
    LANGUAGES, FEATURE_CHAT, FEATURE_VOICE, FEATURE_FIR,
    FEATURE_COURT, FEATURE_VOTER, FEATURE_BILLING, FEATURE_MISSING_PERSON,
)
from corpus_migration import run_corpus_migration
from push import register_device, unregister_device
from push_jobs import run_push_jobs_loop
from account_deletion import hard_delete_user
from script_guard import check_script_mismatch
from personal_law import (
    classify_personal_law_topic,
    detect_context as detect_personal_law_context,
    act_hint_phrase,
    disambiguation_question,
    act_disclaimer,
)

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

# QA Outbound Guard — refuse to boot without an explicit OUTBOUND_MODE.
from services.outbound_guard import validate_outbound_config, outbound_live, outbound_mode, policy_snapshot  # noqa: E402
from services.login_guard import ensure_indexes as _login_guard_indexes  # noqa: E402
validate_outbound_config()

MONGO_URL = os.environ["MONGO_URL"]
DB_NAME = os.environ["DB_NAME"]
CORPUS_DB_NAME = os.environ.get("CORPUS_DB_NAME", "bns-know-your-rights-gandhikar_db")
EMERGENT_LLM_KEY = os.environ["EMERGENT_LLM_KEY"]
JWT_SECRET = os.environ["JWT_SECRET"]
STRIPE_API_KEY = os.environ.get("STRIPE_API_KEY", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
RAZORPAY_KEY_ID = os.environ.get("RAZORPAY_KEY_ID", "")
RAZORPAY_KEY_SECRET = os.environ.get("RAZORPAY_KEY_SECRET", "")
RAZORPAY_WEBHOOK_SECRET = os.environ.get("RAZORPAY_WEBHOOK_SECRET", "")
RAZORPAY_ME_HANDLE = os.environ.get("RAZORPAY_ME_HANDLE", "calviltech")
RAZORPAY_ME_URL = os.environ.get("RAZORPAY_ME_URL", "https://razorpay.me/@calviltech")

# ─── eCourtsIndia partner API ──────────────────────────────────────────────────
ECOURTS_TOKEN = os.getenv("ECOURTS_API_TOKEN", "")
ECOURTS_BASE  = os.getenv("ECOURTS_API_BASE", "https://webapi.ecourtsindia.com")

# ── Server-side LLM config — never expose to clients ──────────────────────────
# These values are authoritative; any model_provider/model_name sent by the
# client is silently ignored (see Fix 2 in the P0 security sprint).
SERVER_CHAT_PROVIDER = os.getenv("CHAT_PROVIDER", "anthropic")
SERVER_CHAT_MODEL    = os.getenv("CHAT_MODEL",    "claude-sonnet-4-5-20250929")
PRO_PRICE_INR = int(os.environ.get("PRO_PRICE_INR", "9900"))  # paise (₹99)
PRO_PRICE_LABEL = os.environ.get("PRO_PRICE_LABEL", "₹99")
PRO_PRICE_USD = int(os.environ.get("PRO_PRICE_USD", "500"))  # cents
PRO_PRICE_USD_LABEL = os.environ.get("PRO_PRICE_USD_LABEL", "$5")
PRO_FREE_SAMPLES = int(os.environ.get("PRO_FREE_SAMPLES", "5"))
# Ready-to-send notice drafts: the first one is free, the rest are a Pro feature.
DRAFTS_FREE = int(os.environ.get("DRAFTS_FREE", "1"))

# ---------------------------------------------------------------------------
# LLM spend caps. Voice is an accessibility floor (not a premium feature), so
# free users get ONE unified daily bucket of 30 queries covering both text and
# voice together. Pro users are unlimited on a per-user basis (only the
# app-wide backstop applies to them).
# ---------------------------------------------------------------------------
FREE_DAILY_QUERIES = int(os.environ.get("FREE_DAILY_QUERIES", "30"))   # unified text + voice
APP_DAILY_LLM_CALLS = int(os.environ.get("APP_DAILY_LLM_CALLS", "3000"))
# Legacy env vars kept for backward compat but no longer used for limits.
_LEGACY_FREE_Q = int(os.environ.get("FREE_DAILY_QUESTIONS", "30"))
_LEGACY_FREE_V = int(os.environ.get("FREE_DAILY_VOICE", "9999"))
JWT_ALG = "HS256"
JWT_EXP_DAYS = 30

if STRIPE_API_KEY:
    stripe.api_key = STRIPE_API_KEY

razor_client = None
if razorpay and RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET:
    try:
        razor_client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))
    except Exception:
        razor_client = None

app = FastAPI(title="Dhara API")

# ── Global JSON error guarantees ──────────────────────────────────────────────
# Ensure every error response from this app is application/json — never HTML.
# This prevents "Unexpected non-whitespace character after JSON" on the client
# when a proxy, middleware, or unhandled exception would otherwise produce a
# plain-text or HTML body that the frontend's r.json() cannot parse.

@app.exception_handler(RequestValidationError)
async def _validation_error_handler(request: Request, exc: RequestValidationError):
    """Return Pydantic v2 validation errors as a clean JSON object.

    Converts the list-of-errors detail format to a human-readable string so
    the client can display it directly without special-casing arrays.
    CORS headers are added explicitly because this handler short-circuits
    CORSMiddleware and cross-origin clients would otherwise be blocked.
    """
    errors = exc.errors()
    first = errors[0] if errors else {}
    field = (first.get("loc") or [])[-1] if first.get("loc") else ""
    msg   = first.get("msg") or "Invalid request"
    detail = f"{field}: {msg}" if field and isinstance(field, str) else msg
    # Sanitise errors for JSON: Pydantic v2 ctx.error may hold a raw Exception
    def _sanitise(obj):
        if isinstance(obj, dict):
            return {k: _sanitise(v) for k, v in obj.items() if k != "url"}
        if isinstance(obj, (list, tuple)):
            return [_sanitise(i) for i in obj]
        if isinstance(obj, Exception):
            return str(obj)
        try:
            import json as _json; _json.dumps(obj); return obj
        except (TypeError, ValueError):
            return str(obj)
    safe_errors = _sanitise(errors)
    return JSONResponse(
        status_code=422,
        content={"detail": detail, "errors": safe_errors},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, PATCH",
            "Access-Control-Allow-Headers": "*",
        },
    )


def _record_error_event(request: Request, exc: Exception) -> None:
    """Persist a privacy-safe record of an unhandled server error.

    Fire-and-forget: scheduled as a task so it never delays the response, and
    fully wrapped so a database problem can never turn into a second failure.

    PRIVACY — the record holds the route, method, error type, status code, a
    trimmed traceback and a timestamp. Never the request body, headers, tokens,
    query string, coordinates, message contents, phone numbers or e-mails.
    """
    import asyncio as _asyncio
    import traceback as _tb
    from datetime import datetime as _dt, timezone as _tz

    try:
        tb = "".join(_tb.format_exception(type(exc), exc, exc.__traceback__))[-4000:]
        doc = {
            "route": request.url.path,          # path only — query string dropped
            "method": request.method,
            "error_type": type(exc).__name__,
            "traceback": tb,
            "status_code": 500,
            "created_at": _dt.now(_tz.utc),
        }

        async def _save():
            try:
                await db.error_events.insert_one(doc)
            except Exception:
                pass  # never let error logging raise

        _asyncio.get_running_loop().create_task(_save())
    except Exception:
        pass


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception):
    """Catch-all: guarantees JSON even when an unhandled exception escapes
    FastAPI's own router (e.g. during startup hooks, ASGI middleware, or
    database connection failures). Logs the full traceback server-side.
    CORS headers are added explicitly because this handler short-circuits
    CORSMiddleware and cross-origin clients would otherwise be blocked.
    """
    import logging as _log
    _log.getLogger("gandhikar").exception(
        "Unhandled exception on %s: %s", request.url.path, exc
    )
    _record_error_event(request, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again."},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, PATCH",
            "Access-Control-Allow-Headers": "*",
        },
    )

api = APIRouter(prefix="/api")

# ── Modular routers (Phase 1 extraction) ─────────────────────────────────────
# db/corpus_db/current_user and shared utilities live in dependencies.py (single connection)
from dependencies import db, corpus_db, current_user, current_user_optional, hash_pw, check_pw, make_token, public_user, meter_llm_use, build_system_prompt, translate_for_retrieval, logger, check_admin_key as _check_admin_key  # noqa: E501
from api.auth_router     import router as auth_router
from api.chat_router     import router as chat_router, whisper_transcribe, read_validated_audio
from api.fir_router      import router as fir_router
from api.visitor_router  import router as visitor_router
from api.journeys_router import router as journeys_router
from api.women_router    import router as women_router
from api.evidence_vault_router import router as evidence_vault_router, cleanup_stale_upload_sessions
from api.scan_router     import router as scan_router
from api.issues_router   import router as issues_router, ensure_indexes as _ensure_issue_indexes
from api.safety_router   import router as safety_router   # GO1A — saved places
from api.geocode_router  import router as geocode_router   # GO1A — address search proxy
from api.geo_router      import router as geo_router        # Google Places (New) autocomplete pilot
from api.journey_router  import router as journey_router  # S3 Phase 2 — guardian status
from api.profile_router  import router as profile_router         # P1 — profile photo
from api.trusted_circle_router import router as trusted_circle_router  # P3 — recipient invitations
from api.alerts_router   import router as alerts_router           # P3/P4 — safety alerts inbox
from api.upi_billing_router import router as upi_billing_router   # UPI payment page (outside the app)
api.include_router(auth_router)
api.include_router(chat_router)
api.include_router(fir_router)
api.include_router(visitor_router)
api.include_router(journeys_router)
api.include_router(women_router)
api.include_router(safety_router)   # GO1A — saved places
api.include_router(geocode_router)  # GO1A — address search proxy
api.include_router(geo_router)      # Google Places (New) autocomplete pilot (flagged off by default)
api.include_router(journey_router)  # S3 Phase 2 — guardian status
api.include_router(evidence_vault_router)
api.include_router(scan_router)
api.include_router(issues_router)   # In-app issue reporter (testers + users)
api.include_router(profile_router)
api.include_router(trusted_circle_router)
api.include_router(alerts_router)
api.include_router(upi_billing_router)

# ---------------------------------------------------------------------------
# Build-only fix (live prod bug, 2025-06 — stale-client JSON error):
# some deployed client builds call /auth/signup, /auth/login, /auth/session
# WITHOUT the /api/ prefix. On this platform's ingress, paths without /api/
# route to the frontend (port 3000) and return an HTML page, which the
# client then fails to JSON.parse. These three aliases are registered on
# `app` directly (not on `api`, so no /api prefix) to match that exact path,
# and just forward straight into the existing /api/auth/* handlers below —
# zero duplicated logic, zero behaviour change, always JSON (FastAPI's
# default error handlers are JSON too, never HTML).
from fastapi import Request as _AliasRequest
from api.auth_router import (
    RegisterIn as _AliasRegisterIn, LoginIn as _AliasLoginIn, GoogleSessionIn as _AliasGoogleSessionIn,
    AuthOut as _AliasAuthOut,
    register as _alias_register_fn, login as _alias_login_fn, google_auth_session as _alias_google_session_fn,
)

@app.post("/auth/signup", response_model=_AliasAuthOut)
async def _alias_auth_signup(body: _AliasRegisterIn):
    return await _alias_register_fn(body)

@app.post("/auth/login", response_model=_AliasAuthOut)
async def _alias_auth_login(body: _AliasLoginIn, request: _AliasRequest):
    # login() requires the Request for B-16 brute-force IP tracking; forward it.
    return await _alias_login_fn(body, request)

@app.post("/auth/session", response_model=_AliasAuthOut)
async def _alias_auth_session(body: _AliasGoogleSessionIn):
    return await _alias_google_session_fn(body)



@app.get("/health")
async def root_health():
    """Root-level liveness probe — no DB, no AI, always fast."""
    return {"status": "ok"}


@app.get("/health/ready")
async def health_ready():
    """Readiness probe — checks every sub-system and reports PASS/FAIL per component.
    Returns 200 even when some components are degraded (Kubernetes pattern).
    Returns 503 only when the core database is unreachable.
    """
    import asyncio
    from config.settings import (
        EMERGENT_LLM_KEY, MONGO_URL, ECOURTS_TOKEN,
        FEATURE_CHAT, FEATURE_VOICE, FEATURE_FIR, FEATURE_COURT,
        FEATURE_VOTER, FEATURE_BILLING, SERVER_CHAT_MODEL, SERVER_CHAT_PROVIDER,
    )

    report = {}
    overall_ok = True

    # 1. Database
    try:
        await asyncio.wait_for(db.command("ping"), timeout=3.0)
        report["database"] = {"status": "PASS", "detail": "MongoDB responsive"}
    except Exception as e:
        report["database"] = {"status": "FAIL", "detail": str(e)[:120]}
        overall_ok = False

    # 2. Corpus DB
    try:
        count = await asyncio.wait_for(corpus_db.legal_sections.estimated_document_count(), timeout=3.0)
        report["corpus_db"] = {"status": "PASS", "detail": f"{count:,} sections indexed"}
    except Exception as e:
        report["corpus_db"] = {"status": "DEGRADED", "detail": str(e)[:120]}

    # 3. LLM / AI Gateway
    if FEATURE_CHAT:
        if EMERGENT_LLM_KEY:
            report["ai_primary"] = {
                "status": "PASS",
                "detail": f"key present provider={SERVER_CHAT_PROVIDER} model={SERVER_CHAT_MODEL}",
            }
        else:
            report["ai_primary"] = {"status": "FAIL", "detail": "EMERGENT_LLM_KEY missing"}
            overall_ok = False
    else:
        report["ai_primary"] = {"status": "DISABLED", "detail": "FEATURE_CHAT=false"}

    # 4. Voice
    report["voice"] = {
        "status": "PASS" if FEATURE_VOICE else "DISABLED",
        "detail": "expo-audio + OpenAI TTS" if FEATURE_VOICE else "FEATURE_VOICE=false",
    }

    # 5. eCourts
    from services.court_data import engine_status as _court_status
    from services.alerts import alerts_configured as _alerts_status
    _cs = _court_status()
    report["court_service"] = {
        "status": "PASS" if (FEATURE_COURT and _cs["configured"]) else (
            "DEGRADED" if FEATURE_COURT else "DISABLED"
        ),
        "detail": (
            "token configured; breaker open" if (_cs["configured"] and _cs["breaker"]["open"])
            else "token configured" if _cs["configured"]
            else "ECOURTS_API_TOKEN not set"
        ),
        "engine": _cs,
        "ops_alerts": _alerts_status(),
    }

    # 6. FIR engine
    try:
        from fir_engine import STAGE_SAFETY_GATE
        report["fir_engine"] = {"status": "PASS" if FEATURE_FIR else "DISABLED", "detail": "importable"}
    except Exception as e:
        report["fir_engine"] = {"status": "FAIL", "detail": str(e)[:120]}

    # 7. Feature flags summary
    report["feature_flags"] = {
        "status": "PASS",
        "detail": {
            "chat": FEATURE_CHAT, "voice": FEATURE_VOICE, "fir": FEATURE_FIR,
            "court": FEATURE_COURT, "voter": FEATURE_VOTER, "billing": FEATURE_BILLING,
        },
    }

    # 8. Required env vars
    required_vars = ["MONGO_URL", "DB_NAME", "JWT_SECRET", "EMERGENT_LLM_KEY"]
    missing_vars = [v for v in required_vars if not os.environ.get(v)]
    report["env_vars"] = {
        "status": "PASS" if not missing_vars else "FAIL",
        "detail": "all required vars present" if not missing_vars else f"missing: {missing_vars}",
    }
    if missing_vars:
        overall_ok = False

    status_code = 200 if overall_ok else 503
    from fastapi.responses import JSONResponse
    return JSONResponse(
        content={"ready": overall_ok, "components": report},
        status_code=status_code,
    )


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("gandhikar")

# ── Guardian token log masking ─────────────────────────────────────────────────
# Replaces the capability token in the two guardian routes with [TOKEN] before
# any log record is emitted, so tokens never appear in supervisor log files.
#   GET /api/journey/status/ABCDE12345       → .../status/[TOKEN]
#   GET /api/journey/ABCDE12345/guardian     → .../[TOKEN]/guardian
_GUARDIAN_LOG_RE = re.compile(
    r"(/api/journey/(?:status/)?)([A-Za-z0-9_\-]{8,40})((?:/guardian)?)",
    re.IGNORECASE,
)

class _GuardianTokenMaskFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        def _mask(s: str) -> str:
            return _GUARDIAN_LOG_RE.sub(lambda m: f"{m.group(1)}[TOKEN]{m.group(3)}", s)
        if isinstance(record.msg, str):
            record.msg = _mask(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {k: _mask(v) if isinstance(v, str) else v
                               for k, v in record.args.items()}
            elif isinstance(record.args, (list, tuple)):
                record.args = tuple(_mask(a) if isinstance(a, str) else a
                                    for a in record.args)
        return True

_gtmf = _GuardianTokenMaskFilter()
logging.getLogger("uvicorn.access").addFilter(_gtmf)
logging.getLogger("uvicorn").addFilter(_gtmf)

# ---------- Models ----------
class CheckoutIn(BaseModel):
    return_url: str

class AdvocateRegisterIn(BaseModel):
    user_id: str
    bar_council_number: str          # required — must be a non-empty enrolment number
    state_bar: str
    specializations: List[str]

class IntakeCreateIn(BaseModel):
    advocate_id: str
    template_id: str = "general"
    template_title: str = "General Intake"

class IntakeSubmitIn(BaseModel):
    client_name: str
    transcript: List[str] = []
    answers: Optional[dict] = None  # structured {question_id: answer}

# ---------- Helpers ----------
@api.post("/billing/checkout")
async def create_checkout(body: CheckoutIn, user: dict = Depends(current_user)):
    if not STRIPE_API_KEY:
        raise HTTPException(500, "Stripe not configured")
    if user.get("is_pro"):
        return {"already_pro": True}
    try:
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[{
                "price_data": {
                    "currency": "usd",
                    "product_data": {
                        "name": "Dhara Pro — Lawyer-style AI Consultation",
                        "description": "Deep, structured legal answers with drafts, escalation paths & action plans."
                    },
                    "unit_amount": PRO_PRICE_USD,
                },
                "quantity": 1,
            }],
            mode="payment",
            success_url=body.return_url + ("&" if "?" in body.return_url else "?") + "status=success",
            cancel_url=body.return_url + ("&" if "?" in body.return_url else "?") + "status=cancel",
            client_reference_id=user["id"],
            customer_email=user["email"],
            metadata={"user_id": user["id"], "email": user["email"], "product": "dhara_pro"},
        )
        # Log intent
        await db.billing_intents.insert_one({
            "id": str(uuid.uuid4()),
            "user_id": user["id"],
            "provider": "stripe",
            "stripe_session_id": session.id,
            "amount_usd_cents": PRO_PRICE_USD,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "created",
        })
        return {"url": session.url, "session_id": session.id}
    except Exception as e:
        logger.exception("stripe checkout failed")
        raise HTTPException(500, f"Checkout failed: {e}")

async def _mark_pro(user_id: str, session_id: Optional[str] = None, provider: str = "stripe", payment_ref: Optional[str] = None):
    now = datetime.now(timezone.utc).isoformat()
    await db.users.update_one(
        {"id": user_id},
        {"$set": {"is_pro": True, "pro_since": now, "pro_provider": provider, "pro_payment_ref": payment_ref}},
    )
    filter_q: dict
    if provider == "stripe" and session_id:
        filter_q = {"stripe_session_id": session_id}
    elif provider == "razorpay" and payment_ref:
        filter_q = {"razorpay_payment_id": payment_ref}
    else:
        filter_q = {"user_id": user_id, "status": "created"}
    await db.billing_intents.update_one(
        filter_q,
        {"$set": {"status": "paid", "paid_at": now, "provider": provider}},
    )

@api.post("/billing/verify")
async def verify_checkout(payload: dict, user: dict = Depends(current_user)):
    """Fallback verification: frontend calls this after returning from Stripe to fast-track Pro flip.
    (Webhook is the source of truth; this only confirms if session is paid.)"""
    sid = payload.get("session_id")
    if not sid or not STRIPE_API_KEY:
        raise HTTPException(400, "session_id required")
    try:
        s = stripe.checkout.Session.retrieve(sid)
        if s.payment_status == "paid" and s.client_reference_id == user["id"]:
            await _mark_pro(user["id"], session_id=sid, provider="stripe")
            return {"is_pro": True}
        return {"is_pro": bool(user.get("is_pro"))}
    except Exception as e:
        logger.exception("verify failed")
        raise HTTPException(500, str(e))

@app.post("/api/webhooks/stripe")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig = request.headers.get("stripe-signature", "")
    # Fail closed. Without a configured signing secret ANY caller could POST a
    # fake "checkout.session.completed" and unlock Pro for any user id, so an
    # unsigned webhook is never processed.
    if not STRIPE_WEBHOOK_SECRET:
        logger.error("stripe webhook rejected: STRIPE_WEBHOOK_SECRET is not configured")
        raise HTTPException(503, "Webhook not configured")
    try:
        event = stripe.Webhook.construct_event(payload, sig, STRIPE_WEBHOOK_SECRET)
    except Exception:
        logger.warning("stripe webhook rejected: bad signature")
        raise HTTPException(400, "Invalid webhook signature")

    et = event["type"] if isinstance(event, dict) else event.get("type")
    if et == "checkout.session.completed":
        session = (event["data"]["object"] if isinstance(event, dict) else event.data.object)
        user_id = session.get("client_reference_id") if isinstance(session, dict) else session.client_reference_id
        sid = session.get("id") if isinstance(session, dict) else session.id
        if user_id:
            await _mark_pro(user_id, session_id=sid, provider="stripe")
    return {"ok": True}

# ---------- Razorpay ----------
class RazorpayVerifyIn(BaseModel):
    payment_id: str = Field(min_length=4, max_length=64)

@api.get("/billing/razorpay/config")
async def razorpay_config():
    return {
        # Only advertise the India payment path when the server can actually
        # verify a payment. Without keys we cannot prove anyone paid, so the
        # option is hidden rather than granted on trust.
        "enabled": bool(razor_client),
        "handle": RAZORPAY_ME_HANDLE,
        "link_url": RAZORPAY_ME_URL,
        "key_id_public": RAZORPAY_KEY_ID or None,
        "server_verify": bool(razor_client),
        "webhook_enabled": bool(RAZORPAY_WEBHOOK_SECRET),
        "amount_paise": PRO_PRICE_INR,
        "amount_label": PRO_PRICE_LABEL,
        "currency": "INR",
    }

@api.post("/billing/razorpay/submit-payment-id")
async def razorpay_submit_payment(body: RazorpayVerifyIn, user: dict = Depends(current_user)):
    """User pastes the Razorpay Payment ID (pay_xxx) shown after paying.

    FAIL CLOSED: Pro is granted only when Razorpay itself confirms the payment is
    captured/authorised for at least the Pro price. The old build fell back to
    trusting the pasted id whenever server keys were missing, which let anyone
    unlock Pro for free by typing "pay_" followed by anything.
    """
    if user.get("is_pro"):
        return {"is_pro": True, "already_pro": True}

    if not razor_client:
        raise HTTPException(
            503,
            "Online payment is temporarily unavailable. Please try again later.",
        )

    pid = body.payment_id.strip()
    if not pid.startswith("pay_"):
        raise HTTPException(400, "Payment ID must start with pay_")

    now = datetime.now(timezone.utc).isoformat()
    intent_doc = {
        "id": str(uuid.uuid4()),
        "user_id": user["id"],
        "email": user["email"],
        "provider": "razorpay",
        "razorpay_payment_id": pid,
        "amount_paise": PRO_PRICE_INR,
        "status": "submitted",
        "created_at": now,
    }

    verified = False
    verify_error: Optional[str] = None

    try:
        payment = razor_client.payment.fetch(pid)
        intent_doc["razorpay_payment"] = {
            "status": payment.get("status"),
            "amount": payment.get("amount"),
            "currency": payment.get("currency"),
            "email": payment.get("email"),
            "contact": payment.get("contact"),
            "method": payment.get("method"),
        }
        status = payment.get("status")
        amount = int(payment.get("amount") or 0)
        if status in ("captured", "authorized") and amount >= PRO_PRICE_INR:
            verified = True
        else:
            verify_error = f"Payment status={status}, amount={amount} paise (required {PRO_PRICE_INR})"
    except Exception as e:
        logger.exception("razorpay fetch failed")
        verify_error = str(e)[:200]

    # Duplicate protection: a payment_id should only unlock Pro once
    existing = await db.billing_intents.find_one({"razorpay_payment_id": pid, "status": "paid"})
    if existing and existing.get("user_id") != user["id"]:
        raise HTTPException(400, "This payment has already been used by another account.")

    if verified:
        intent_doc["status"] = "paid"
        intent_doc["paid_at"] = now
        await db.billing_intents.insert_one(intent_doc)
        await _mark_pro(user["id"], provider="razorpay", payment_ref=pid)
        return {"is_pro": True, "verified": True}
    else:
        intent_doc["status"] = "verify_failed"
        intent_doc["verify_error"] = verify_error
        await db.billing_intents.insert_one(intent_doc)
        raise HTTPException(400, "We could not confirm that payment yet. Please try again in a few minutes.")

@app.post("/api/webhooks/razorpay")
async def razorpay_webhook(request: Request):
    """Configure this URL in Razorpay Dashboard → Settings → Webhooks.
    Event: payment.captured. Set the webhook secret and put it in RAZORPAY_WEBHOOK_SECRET."""
    payload = await request.body()
    sig = request.headers.get("x-razorpay-signature", "")

    # Fail closed — an unsigned webhook is an unauthenticated "make this user
    # Pro" endpoint, so it is rejected when no secret is configured.
    if not RAZORPAY_WEBHOOK_SECRET:
        logger.error("razorpay webhook rejected: RAZORPAY_WEBHOOK_SECRET is not configured")
        raise HTTPException(503, "Webhook not configured")
    expected = hmac.new(
        RAZORPAY_WEBHOOK_SECRET.encode(),
        payload,
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(expected, sig):
        logger.warning("razorpay webhook rejected: bad signature")
        raise HTTPException(400, "Invalid signature")

    try:
        event = json.loads(payload)
    except Exception:
        raise HTTPException(400, "Invalid JSON")

    et = event.get("event")
    if et in ("payment.captured", "payment.authorized"):
        payment = (event.get("payload") or {}).get("payment", {}).get("entity", {})
        pid = payment.get("id")
        email = (payment.get("email") or "").lower()
        contact = payment.get("contact") or ""
        amount = int(payment.get("amount") or 0)
        # Find user by email or phone (razorpay.me collects both)
        user = None
        if email:
            user = await db.users.find_one({"email": email})
        if not user and contact:
            # try last 10 digits of phone
            digits = "".join(ch for ch in contact if ch.isdigit())[-10:]
            if digits:
                user = await db.users.find_one({"phone": {"$regex": digits + "$"}})
        if user and amount >= PRO_PRICE_INR:
            # Record intent if missing
            existing = await db.billing_intents.find_one({"razorpay_payment_id": pid})
            if not existing:
                await db.billing_intents.insert_one({
                    "id": str(uuid.uuid4()),
                    "user_id": user["id"],
                    "email": user["email"],
                    "provider": "razorpay",
                    "razorpay_payment_id": pid,
                    "amount_paise": amount,
                    "status": "paid_via_webhook",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "paid_at": datetime.now(timezone.utc).isoformat(),
                    "razorpay_payment": payment,
                })
            await _mark_pro(user["id"], provider="razorpay", payment_ref=pid)
            logger.info(f"razorpay webhook: marked user {user['email']} pro via {pid}")
    return {"ok": True}


# LANGUAGES imported from config.settings

MODELS = [
    {"provider": "anthropic", "name": "claude-sonnet-4-5-20250929", "label": "Dhara AI", "recommended": True},
    {"provider": "openai", "name": "gpt-5.2", "label": "GPT-5.2"},
    {"provider": "gemini", "name": "gemini-3.1-pro-preview", "label": "Gemini 3.1 Pro"},
]

from rights_topics_data import TOPICS, localize_topic  # noqa: E402

@api.get("/reference/languages")
async def get_languages():
    return LANGUAGES

@api.get("/reference/states")
async def get_states():
    return STATES

@api.get("/reference/models")
async def get_models():
    return MODELS

# ── /api/retrieve — standalone corpus search ────────────────────────────────
# Searches the MongoDB legal_sections corpus with optional state jurisdiction
# prioritisation.  The answer pipeline (chat/stream) also calls corpus_db
# internally; this endpoint exposes it for external clients and front-end
# debug tooling.
#
# Request body:
#   query          (str, required)  — user's natural-language question
#   state_code     (str, optional)  — 2-letter code e.g. "MH"; boosts state Acts
#   limit          (int, optional)  — max results returned, default 5, max 10
#   mode           (str, optional)  — "text" (default) | "exact"
#   section_number (str, optional)  — required when mode="exact"
#   act_hint       (str, optional)  — act name substring for exact lookup
class RetrieveIn(BaseModel):
    query: str
    state_code: Optional[str] = None
    limit: int = 5
    mode: str = "text"
    section_number: Optional[str] = None
    act_hint: Optional[str] = None

@api.post("/retrieve")
async def retrieve(body: RetrieveIn, user: dict = Depends(current_user)):
    """Search the legal corpus.  Returns up to `limit` matching sections,
    ranked by full-text relevance and boosted by the user's state jurisdiction.

    Safety guards (G1 dead-law, G2 judicial join, G3 badge) are applied
    automatically inside corpus_db.retrieve_db / lookup_section.
    Falls back gracefully if the corpus collection is empty or the text index
    is still building."""
    limit = max(1, min(int(body.limit), 10))

    if body.mode == "exact" and body.section_number:
        result = await db_lookup_section(corpus_db, body.section_number, body.act_hint)
        results = [result] if result else []
    else:
        # Use the user's own state_jurisdiction if the caller didn't specify
        state_code = body.state_code or user.get("state") or None
        try:
            results = await db_retrieve(corpus_db, body.query, state_code=state_code, limit=limit)
        except Exception:
            results = []

    return {
        "query": body.query,
        "count": len(results),
        "results": results,
        "corpus_source": "mongodb",
        "state_priority": (
            STATE_CODE_TO_JURISDICTION.get((body.state_code or user.get("state") or "").upper(), None)
        ),
    }


@api.get("/reference/topics")
async def get_topics(lang: str = Query("en")):
    return [localize_topic(t, lang) for t in TOPICS]

@api.get("/reference/topics/{topic_id}")
async def get_topic(topic_id: str, lang: str = Query("en")):
    for t in TOPICS:
        if t["id"] == topic_id:
            return localize_topic(t, lang)
    raise HTTPException(404, "Topic not found")

# ---------- Legal / meta ----------
@api.get("/legal/terms")
async def get_terms():
    return {
        "version": TERMS_VERSION,
        "text": TERMS_AND_CONDITIONS,
        "disclaimer_short": DISCLAIMER_SHORT,
        "copyright": "© Callistus Moses",
        "company": "Msafe",
    }

@api.get("/billing/pricing")
async def pricing():
    return {
        "pro_price_inr_paise": PRO_PRICE_INR,
        "pro_price_label": PRO_PRICE_LABEL,
        "pro_price_usd_cents": PRO_PRICE_USD,
        "pro_price_usd_label": PRO_PRICE_USD_LABEL,
        "currency": "INR",
        "currency_intl": "USD",
        "billing_type": "one_time",
        "providers": {
            "stripe": {
                "enabled": bool(STRIPE_API_KEY),
                "currency": "USD",
                "amount_label": PRO_PRICE_USD_LABEL,
                "regions": ["Canada", "International"],
            },
            "razorpay": {
                "enabled": bool(razor_client),
                "currency": "INR",
                "amount_label": PRO_PRICE_LABEL,
                "regions": ["India"],
                "link_url": RAZORPAY_ME_URL,
                "handle": RAZORPAY_ME_HANDLE,
            },
        },
        "features": [
            "Lawyer-consultation-style deep answers",
            "Ready-to-use draft complaint / RTI / notice paragraphs",
            "Step-by-step action plans with jurisdiction & authority names",
            "Counter-arguments & pitfalls analysis",
            "Escalation paths (SP, DM, HRC, NALSA, Consumer Commission)",
            "Priority AI models",
        ],
    }

@api.get("/health")
async def health():
    return {
        "status": "ok",
        "app": "Dhara",
        "copyright": "© Callistus Moses",
        "company": "Msafe",
        "terms_version": TERMS_VERSION,
        "outbound_mode": outbound_mode(),
        "outbound": policy_snapshot(),
    }


@api.get("/qa/outbox")
async def qa_outbox(
    to: Optional[str] = None,
    kind: Optional[str] = None,
    limit: int = 20,
    x_admin_key: Optional[str] = Header(None),
):
    """QA-only isolated inbox: simulated emails / pushes recorded by the
    Outbound Guard. Disabled (404) whenever OUTBOUND_MODE=live."""
    if outbound_live():
        raise HTTPException(404, "Not found")
    _check_admin_key(x_admin_key)
    q: dict = {}
    if to:
        q["target"] = to
    if kind:
        q["kind"] = kind
    items = await db.qa_outbox.find(q, {"_id": 0}).sort("created_at", -1).to_list(max(1, min(limit, 100)))
    return {"outbound_mode": outbound_mode(), "items": items}

# ---------- Admin CSV export ----------
# Key-protected read-only export for the app operator. The key MUST be sent in
# the X-Admin-Key header — it used to be a ?key= query parameter, which leaks the
# secret into access logs, proxy logs and browser history for an endpoint that
# dumps every user's PII and chat history.
#   curl -H "X-Admin-Key: $ADMIN_KEY" <API>/api/admin/export/users.csv
# Password hashes are NEVER exported. Rows are streamed so this handles large tables.
# `_check_admin_key` now lives in dependencies.py (shared with other admin
# endpoints, e.g. api/journeys_router.py) — imported above.

@api.get("/admin/export/users.csv")
async def export_users_csv(x_admin_key: Optional[str] = Header(None)):
    _check_admin_key(x_admin_key)
    import csv, io
    from fastapi.responses import StreamingResponse

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow([
        "id", "email", "name", "phone", "language", "is_pro", "pro_since",
        "pro_samples_used", "auto_speak", "terms_version", "terms_accepted_at",
        "created_at", "password_reset_at",
    ])
    async for u in db.users.find({}).sort("created_at", -1):
        writer.writerow([
            u.get("id", ""),
            u.get("email", ""),
            u.get("name", ""),
            u.get("phone", ""),
            u.get("language", ""),
            "yes" if u.get("is_pro") else "no",
            u.get("pro_since", ""),
            u.get("pro_samples_used", 0),
            "yes" if u.get("auto_speak") else "no",
            u.get("terms_version", ""),
            u.get("terms_accepted_at", ""),
            u.get("created_at", ""),
            u.get("password_reset_at", ""),
        ])
    buffer.seek(0)
    return StreamingResponse(
        io.BytesIO(buffer.read().encode("utf-8")),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="dhara_users_{datetime.now(timezone.utc).strftime("%Y%m%d")}.csv"',
        },
    )

@api.get("/admin/export/messages.csv")
async def export_messages_csv(x_admin_key: Optional[str] = Header(None)):
    _check_admin_key(x_admin_key)
    import csv, io
    from fastapi.responses import StreamingResponse

    # P0 containment (T07 — bulk-export privacy defect, 2026-10-26
    # remediation): this endpoint used to join every user's full legal-
    # conversation content with their email/name/phone in one unscoped CSV
    # — exactly the "export combines users' legal conversations with
    # personal details" defect flagged for containment. It is RESTRICTED,
    # not removed: message content remains available for legitimate
    # operational debugging, but is no longer bundled with directly-
    # identifying personal details in the same unscoped dump. `user_id`
    # (an opaque internal id, already present on every message) is kept so
    # messages can still be correlated to an account when a SEPARATE,
    # deliberate lookup is actually needed — that is a materially smaller
    # disclosure than shipping name+email+phone next to 2000 chars of a
    # person's actual legal situation for every user in one file.
    session_index: dict = {}
    async for s in db.sessions.find({}, projection={"id": 1, "user_id": 1}):
        session_index[s["id"]] = s.get("user_id", "")

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow([
        "timestamp", "user_id", "role",
        "status", "content", "citations", "session_id",
    ])
    async for m in db.messages.find({}).sort("created_at", -1):
        sid = m.get("session_id", "")
        uid = m.get("user_id") or session_index.get(sid, "")
        citations = m.get("citations") or []
        # Flatten citation labels for CSV
        cit_labels = " | ".join(
            (c.get("short_label", "") if isinstance(c, dict) else str(c)) for c in citations
        )
        # Derive status: prefer explicit field, else infer from error / content
        status = m.get("status")
        if not status:
            if m.get("error"):
                status = "error"
            elif m.get("role") == "assistant" and m.get("content", "").startswith(
                ("I don't have a verified source", "I cannot confirm this from the official text", "I only cover Indian law", "I can only help")
            ):
                status = "refused"
            else:
                status = "ok"
        writer.writerow([
            m.get("timestamp") or m.get("created_at", ""),
            uid,
            m.get("role", ""),
            status,
            (m.get("content") or "").replace("\r", " ").replace("\n", " ")[:2000],
            cit_labels,
            sid,
        ])
    buffer.seek(0)
    return StreamingResponse(
        io.BytesIO(buffer.read().encode("utf-8")),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="dhara_messages_{datetime.now(timezone.utc).strftime("%Y%m%d")}.csv"',
        },
    )

@api.get("/admin/stats")
async def admin_stats(x_admin_key: Optional[str] = Header(None)):
    """Quick JSON overview — total users, pro users, session/message counts."""
    _check_admin_key(x_admin_key)
    return {
        "users_total": await db.users.count_documents({}),
        "users_pro": await db.users.count_documents({"is_pro": True}),
        "sessions_total": await db.sessions.count_documents({}),
        "messages_total": await db.messages.count_documents({}),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


@api.get("/admin/refusal-stats")
async def admin_refusal_stats(
    days: int = 7,
    x_admin_key: Optional[str] = Header(None),
):
    """
    Refusal cause-code breakdown (A4 instrumentation) — aggregate-only,
    no user-identifying data is stored or returned. Answers exactly the
    four questions corpus-spend decisions depend on:
      1. no_candidate_scored — plausible true coverage gap (topic not in corpus)
      2. below_confidence_threshold — a candidate existed but scored under the
         bar; a retrieval-tuning question, NOT necessarily a missing law
      3. query_too_vague — too few content words to score anything
      4. non_indian_jurisdiction / not_legal_advice_request — out of scope,
         not a corpus gap at all
    `top_key` counts show WHICH existing entries keep almost-matching (useful
    for keyword tuning) and `category` counts show which real-world topics
    are being asked about, including topics with zero corpus coverage today.
    """
    _check_admin_key(x_admin_key)
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    cursor = db.refusal_events.find({"created_at": {"$gte": since}})
    events = await cursor.to_list(10000)

    total = len(events)
    by_cause: dict = {}
    by_sub_cause: dict = {}
    by_category: dict = {}
    by_language: dict = {}
    score_histogram = {"0": 0, "1-2": 0, "3-5": 0, "6-11": 0, "12+": 0, "n/a": 0}
    near_miss_top_keys: dict = {}  # top_key counts for sub_cause == below_confidence_threshold

    for e in events:
        cause = e.get("cause_code") or "unknown"
        by_cause[cause] = by_cause.get(cause, 0) + 1
        sub = e.get("sub_cause")
        if sub:
            by_sub_cause[sub] = by_sub_cause.get(sub, 0) + 1
        cat = e.get("category") or "other_uncategorized"
        by_category[cat] = by_category.get(cat, 0) + 1
        lang = e.get("language") or "unknown"
        by_language[lang] = by_language.get(lang, 0) + 1

        score = e.get("top_score")
        if score is None:
            score_histogram["n/a"] += 1
        elif score == 0:
            score_histogram["0"] += 1
        elif score <= 2:
            score_histogram["1-2"] += 1
        elif score <= 5:
            score_histogram["3-5"] += 1
        elif score <= 11:
            score_histogram["6-11"] += 1
        else:
            score_histogram["12+"] += 1

        if sub == "below_confidence_threshold" and e.get("top_key"):
            k = e["top_key"]
            near_miss_top_keys[k] = near_miss_top_keys.get(k, 0) + 1

    return {
        "window_days": days,
        "total_refusals": total,
        "by_cause_code": by_cause,
        "by_sub_cause": by_sub_cause,
        "by_category": dict(sorted(by_category.items(), key=lambda x: x[1], reverse=True)),
        "by_language": by_language,
        "score_histogram": score_histogram,
        "near_miss_top_keys": dict(sorted(near_miss_top_keys.items(), key=lambda x: x[1], reverse=True)),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


@api.get("/admin/overview")
async def admin_overview(
    tz: str = "Asia/Kolkata",
    days: int = 30,
    x_admin_key: Optional[str] = Header(None),
):
    """
    Read-only analytics overview — no PII, aggregated counts only.

    Fields:
      registered_accounts  total real accounts (guests, test-email and
                           deleted accounts excluded)
      new_registrations    real accounts whose created_at >= now - days
      dau                  distinct active users today (local tz)
      mau                  distinct active users in the last 30 days
      open_tickets         grievance_tickets not yet in a closed state
      as_of                UTC timestamp of this response

    DAU / MAU source: usage_daily (scope="user", distinct user_id per day).
    This collection is written at query time and deduplicates within the day,
    making it more accurate than counting distinct user_id in messages.

    Guest exclusion:    is_guest != true
    Test exclusion:     email does not match ^TEST_ / ^safety_test_ / @guest.dhara.local
    Deleted exclusion:  hard-delete only — no soft-delete field exists in this
                        schema; deleted users are absent from the collection.
    Closed ticket set:  ["resolved", "closed", "rejected"]
    """
    _check_admin_key(x_admin_key)

    # ── Timezone resolution ───────────────────────────────────────────────────
    try:
        tz_obj = ZoneInfo(tz)
    except (ZoneInfoNotFoundError, KeyError):
        tz_obj = ZoneInfo("Asia/Kolkata")
        tz = "Asia/Kolkata"

    now_utc   = datetime.now(timezone.utc)
    now_local = now_utc.astimezone(tz_obj)

    # usage_daily uses YYYY-MM-DD in local time
    today_local = now_local.date().isoformat()
    mau_since_local = (now_local - timedelta(days=30)).date().isoformat()

    # created_at in users/messages is stored as a naive UTC ISO string
    # (e.g. "2026-07-22T22:14:29.835"). Strip tzinfo before comparing.
    since_utc_naive = (now_utc.replace(tzinfo=None) - timedelta(days=days)).isoformat()

    # ── Exclusion filter for real accounts ────────────────────────────────────
    _real_account = {
        "is_guest": {"$ne": True},
        "email": {
            "$not": {
                "$regex": r"^TEST_|^safety_test_|@guest\.dhara\.local",
                "$options": "i",
            }
        },
    }

    # ── 1. registered_accounts ────────────────────────────────────────────────
    registered_accounts = await db.users.count_documents(_real_account)

    # ── 2. new_registrations (last `days` days) ───────────────────────────────
    new_registrations = await db.users.count_documents({
        **_real_account,
        "created_at": {"$gte": since_utc_naive},
    })

    # ── 3. DAU — distinct user_id active today (local tz) ────────────────────
    dau_pipeline = [
        {"$match": {"scope": "user", "day": today_local}},
        {"$group": {"_id": "$user_id"}},
        {"$count": "n"},
    ]
    dau_result = await db.usage_daily.aggregate(dau_pipeline).to_list(1)
    dau = dau_result[0]["n"] if dau_result else 0

    # ── 4. MAU — distinct user_id active in last 30 days (local tz) ──────────
    mau_pipeline = [
        {"$match": {"scope": "user", "day": {"$gte": mau_since_local}}},
        {"$group": {"_id": "$user_id"}},
        {"$count": "n"},
    ]
    mau_result = await db.usage_daily.aggregate(mau_pipeline).to_list(1)
    mau = mau_result[0]["n"] if mau_result else 0

    # ── 5. open_tickets ───────────────────────────────────────────────────────
    _closed_statuses = ["resolved", "closed", "rejected"]
    open_tickets = await db.grievance_tickets.count_documents(
        {"status": {"$nin": _closed_statuses}}
    )

    return {
        "registered_accounts": registered_accounts,
        "new_registrations":   new_registrations,
        "dau":                 dau,
        "mau":                 mau,
        "open_tickets":        open_tickets,
        "as_of":               now_utc.isoformat(),
        "_meta": {
            "tz":              tz,
            "days":            days,
            "dau_source":      "usage_daily",
            "mau_source":      "usage_daily",
            "today_local":     today_local,
            "mau_since_local": mau_since_local,
            "since_utc_naive": since_utc_naive,
            "closed_statuses": _closed_statuses,
            "guest_field":     "is_guest",
            "test_email_regex": r"^TEST_|^safety_test_|@guest\.dhara\.local",
            "deleted_field":   "none — hard-delete only",
        },
    }


@api.get("/")
async def root():
    return {
        "message": "Dhara API - Empowering every Indian citizen with knowledge of their rights.",
        "copyright": "© Callistus Moses",
        "company": "Msafe",
    }

# ---------- Static downloads (App summary docs) ----------
from fastapi.responses import FileResponse

DOWNLOADS_DIR = Path("/app/downloads")

@api.get("/downloads/{filename}")
async def get_download(filename: str):
    # Allow only .pdf/.docx from the downloads dir, no traversal
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(400, "Invalid filename")
    if not (filename.endswith(".pdf") or filename.endswith(".docx")):
        raise HTTPException(400, "Only .pdf and .docx are downloadable")
    fp = DOWNLOADS_DIR / filename
    if not fp.exists():
        raise HTTPException(404, "File not found")
    media_type = (
        "application/pdf" if filename.endswith(".pdf")
        else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    return FileResponse(str(fp), media_type=media_type, filename=filename)


# ─── eCourts Case Lookup (Dhara Lookup tab) ───────────────────────────────────
# Actual logic lives in services/court_data.py — this file only exposes the
# thin route wrapper so the eCourts partner cannot drag the rest of the API
# down when its SLA slips. See services/court_data.py::CourtDataResult for
# the failure envelope.
from services.court_data import (
    case_by_cnr as _court_case_by_cnr,
    search_by_party as _court_search_by_party,
    engine_status as court_engine_status,
)


def _raise_from_court_result(result) -> None:
    """Translate a non-OK CourtDataResult into an HTTPException with a stable
    correlation ref that also appears in the server log. Keeps the caller-side
    contract identical to the pre-refactor behaviour for clients."""
    detail = result.message
    if result.error_id and result.status != "invalid":
        detail = f"{detail} (ref {result.error_id})"
    raise HTTPException(status_code=result.http_hint or 502, detail=detail)


@api.get("/cases/cnr/{cnr}", summary="Look up a court case by CNR number")
async def case_by_cnr(cnr: str, user: dict = Depends(current_user)):
    result = await _court_case_by_cnr(cnr)
    if result.is_ok():
        return result.data
    if result.status == "not_found":
        raise HTTPException(404, "No matching case was found.")
    _raise_from_court_result(result)


@api.get("/cases/search", summary="Search court cases by party name")
async def search_cases(name: str, page: int = 1, user: dict = Depends(current_user)):
    result = await _court_search_by_party(name, page)
    if result.is_ok():
        return result.data
    _raise_from_court_result(result)

# ─────────────────────────────────────────────────────────────────────────────

# ── Advocate Door — Sprint 5 ─────────────────────────────────────────────────

async def _generate_intake_summary(client_name: str, situation: str, outcome: str) -> tuple[str, str]:
    """Generate a structured advocate brief from client intake transcript using LLM."""
    try:
        system_message = (
            "You are a legal case briefing assistant for an Indian advocate. "
            "Given a client intake, produce a structured brief with exactly two sections:\n"
            "SUMMARY: 2-3 concise sentences summarising the facts and what the client wants.\n"
            "DHARA ANALYSIS: identify the primary legal issue, the relevant area of law "
            "(e.g. family, property, criminal, consumer), and 2-3 recommended next actions "
            "for the advocate (office to visit, document to file, notice to send, etc.).\n"
            "Write in formal but accessible English. No markdown. No section numbers."
        )
        prompt = (
            f"Client name: {client_name}\n\n"
            f"Situation (client's own words):\n{situation}\n\n"
            f"Desired outcome:\n{outcome}\n\n"
            "Write the SUMMARY and DHARA ANALYSIS:"
        )
        # Routed through the AI gateway (model from INTAKE_SUMMARY_SPEC).
        result = await _gateway_send_once(
            f"intake-summary-{uuid.uuid4()}",
            system_message,
            prompt,
            spec=INTAKE_SUMMARY_SPEC,
            caller="advocate_intake_summary",
        )
        full_text = str(result or "").strip()
        # Strip any markdown bold/italic artifacts the model may emit
        import re as _re
        full_text = _re.sub(r'\*{1,3}', '', full_text).strip()
        if "DHARA ANALYSIS" in full_text:
            parts = full_text.split("DHARA ANALYSIS", 1)
            summary = parts[0].replace("SUMMARY:", "").replace("SUMMARY", "").strip()
            analysis = parts[1].lstrip(":").lstrip("**").strip()
        else:
            summary = full_text[:400]
            analysis = full_text[400:].strip()
        return summary, analysis
    except Exception as e:
        logger.warning(f"[intake_summary] LLM call failed: {e}")
        return "", ""


@api.post("/advocate/register")
async def advocate_register(body: AdvocateRegisterIn, user: dict = Depends(current_user)):
    """Register a user as an advocate. Creates an advocate_profiles document."""
    if user["id"] != body.user_id:
        raise HTTPException(403, "user_id mismatch")
    existing = await db.advocate_profiles.find_one({"user_id": body.user_id})
    if existing:
        return {k: v for k, v in existing.items() if k != "_id"}
    # Validate bar council number
    if not body.bar_council_number or len(body.bar_council_number.strip()) < 4:
        raise HTTPException(400, "A valid Bar Council enrolment number is required (min 4 characters)")
    if not body.specializations:
        raise HTTPException(400, "Select at least one specialization")
    now = datetime.now(timezone.utc).isoformat()
    doc = {
        "id": str(uuid.uuid4()),
        "user_id": body.user_id,
        "bar_council_number": body.bar_council_number.strip().upper() if body.bar_council_number else "",
        "state_bar": body.state_bar,
        "specializations": body.specializations[:5],
        "verified": False,
        "created_at": now,
        "updated_at": now,
    }
    try:
        await db.advocate_profiles.insert_one(doc)
    except Exception as e:
        err_str = str(e)
        if "bar_council_number" in err_str:
            raise HTTPException(400, "This Bar Council enrolment number is already registered with another account")
        existing = await db.advocate_profiles.find_one({"user_id": body.user_id})
        if existing:
            return {k: v for k, v in existing.items() if k != "_id"}
        raise HTTPException(500, "Registration failed — please try again")
    await db.verification_log.insert_one({
        "id": str(uuid.uuid4()),
        "advocate_id": body.user_id,
        "bar_council_number": doc["bar_council_number"],
        "state_bar": body.state_bar,
        "action": "registration_submitted",
        "created_at": now,
    })
    return {k: v for k, v in doc.items() if k != "_id"}


@api.get("/advocate/profile/{user_id}")
async def advocate_profile(user_id: str, user: dict = Depends(current_user)):
    """Fetch the advocate profile for a user."""
    if user["id"] != user_id:
        raise HTTPException(403, "Forbidden")
    profile = await db.advocate_profiles.find_one({"user_id": user_id}, {"_id": 0})
    if not profile:
        raise HTTPException(404, "Advocate profile not found")
    return profile


@api.post("/advocate/intake/create")
async def intake_create(request: Request, body: IntakeCreateIn, user: dict = Depends(current_user)):
    """Create a shareable intake link for a client. Returns intake_token and intake_url."""
    if user["id"] != body.advocate_id:
        raise HTTPException(403, "advocate_id mismatch")
    # Verify advocate is registered
    profile = await db.advocate_profiles.find_one({"user_id": body.advocate_id})
    if not profile:
        raise HTTPException(403, "Register as an advocate first")
    token = secrets.token_urlsafe(20)
    now = datetime.now(timezone.utc).isoformat()
    # Build absolute intake URL from the incoming request's origin
    origin = str(request.base_url).rstrip("/")
    intake_url = f"{origin}/intake/{token}"
    doc = {
        "id": str(uuid.uuid4()),
        "intake_token": token,
        "intake_url": intake_url,
        "advocate_id": body.advocate_id,
        "template_id": body.template_id,
        "template_title": body.template_title,
        "client_name": "",
        "status": "pending",
        "transcript": [],
        "answers": {},
        "summary": "",
        "dhara_analysis": "",
        "created_at": now,
        "completed_at": None,
    }
    await db.client_intakes.insert_one(doc)
    return {"intake_token": token, "intake_url": intake_url}


@api.get("/advocate/intake/{token}")
async def intake_get(token: str):
    """Public endpoint — fetch intake form metadata for a given token.
    Used both by the client-facing form and the advocate's intake-view screen."""
    intake = await db.client_intakes.find_one({"intake_token": token}, {"_id": 0})
    if not intake:
        raise HTTPException(404, "Intake not found")
    # For the public client form: only expose status + expiry info (no PII)
    return {
        "id": intake.get("id", ""),
        "intake_token": intake.get("intake_token", ""),
        "template_id": intake.get("template_id", "general"),
        "template_title": intake.get("template_title", "General Intake"),
        "status": intake.get("status", "pending"),
        "client_name": intake.get("client_name", ""),
        "summary": intake.get("summary", ""),
        "dhara_analysis": intake.get("dhara_analysis", ""),
        "transcript": intake.get("transcript", []),
        "answers": intake.get("answers", {}),
        "created_at": intake.get("created_at", ""),
        "completed_at": intake.get("completed_at"),
        "expired": intake.get("status") == "expired",
    }


@api.post("/advocate/intake/{token}/submit")
async def intake_submit(token: str, body: IntakeSubmitIn):
    """Public endpoint — client submits their intake data.
    Saves transcript, then generates an AI summary for the advocate."""
    intake = await db.client_intakes.find_one({"intake_token": token})
    if not intake:
        raise HTTPException(404, "Intake link not found")
    if intake.get("status") in ("complete", "expired"):
        raise HTTPException(400, "This intake link has already been used or has expired")
    now = datetime.now(timezone.utc).isoformat()
    transcript = [t.strip() for t in (body.transcript or []) if t.strip()]
    answers = body.answers or {}
    await db.client_intakes.update_one(
        {"intake_token": token},
        {"$set": {
            "client_name": body.client_name.strip()[:120],
            "transcript": transcript,
            "answers": answers,
            "status": "processing",
            "completed_at": now,
        }},
    )
    # Build context for AI brief from structured answers if available
    if answers:
        situation = "\n".join(f"{k}: {v}" for k, v in answers.items() if v)
        outcome = answers.get("relief", "")
    else:
        situation = transcript[0] if len(transcript) > 0 else ""
        outcome = transcript[1] if len(transcript) > 1 else ""
    summary, analysis = await _generate_intake_summary(body.client_name, situation, outcome)
    await db.client_intakes.update_one(
        {"intake_token": token},
        {"$set": {"summary": summary, "dhara_analysis": analysis, "status": "complete"}},
    )
    return {"ok": True, "message": "Your information has been sent to your advocate securely."}


@api.post("/advocate/intake/{token}/transcribe")
async def intake_transcribe(
    token: str,
    audio: UploadFile = File(...),
    language: str = Form(None),
):
    """Public endpoint — voice-to-text for the client-facing intake form.
    No login exists on this page (it's a shareable link), so this is
    scoped to a live (not yet complete/expired) intake token, and the
    Whisper cost is metered against the advocate who created the link
    (the one who benefits from the client filling the form) rather than
    the anonymous client."""
    intake = await db.client_intakes.find_one({"intake_token": token})
    if not intake:
        raise HTTPException(404, "Intake link not found")
    if intake.get("status") in ("complete", "expired"):
        raise HTTPException(400, "This intake link has already been used or has expired")
    advocate = await db.users.find_one({"id": intake.get("advocate_id")})
    if not advocate:
        raise HTTPException(404, "Advocate account not found")
    clip = await read_validated_audio(audio)   # validate first — rejected files are never billed
    await meter_llm_use(advocate, "voice")
    return await whisper_transcribe(clip, language)


@api.get("/advocate/intakes/{advocate_id}")
async def intakes_list(advocate_id: str, user: dict = Depends(current_user)):
    """Fetch all client intakes created by an advocate (most recent first)."""
    if user["id"] != advocate_id:
        raise HTTPException(403, "Forbidden")
    cursor = db.client_intakes.find({"advocate_id": advocate_id}, {"_id": 0}).sort("created_at", -1).limit(50)
    intakes = await cursor.to_list(length=50)
    return intakes

# ─────────────────────────────────────────────────────────────────────────────

# ── IPC → BNS Cross-Reference Engine ─────────────────────────────────────────

import json as _json_mod
from pathlib import Path as _Path
from engine.approval_evidence import check_section_correspondence_approval as _xref_check_corr

_XREF_PATH = _Path(__file__).parent / "ipc_bns_mapping.json"
_XREF_DATA: dict = {}

def _load_xref(force_reload: bool = False) -> dict:
    global _XREF_DATA
    if not _XREF_DATA or force_reload:
        if _XREF_PATH.exists():
            with open(_XREF_PATH, encoding="utf-8") as _f:
                _XREF_DATA = _json_mod.load(_f)
    return _XREF_DATA


def _xref_search(q: str = "", ipc: str = "", bns: str = "") -> list[dict]:
    """
    Search the IPC↔BNS cross-reference file.
    Matches on:
      - Exact/partial IPC section (ipc param or q starts with a digit)
      - Exact/partial BNS section (bns param)
      - Keyword search in offence + what_changed fields
    Returns a normalised list of result dicts, deduped, max 25.
    """
    data = _load_xref()
    q_lower = q.strip().lower()
    ipc_q  = (ipc or "").strip().upper()
    bns_q  = (bns or "").strip().upper()

    results: list[dict] = []

    def _matches(entry: dict) -> bool:
        # Normalise fields — guard against JSON null values
        def _s(v): return str(v).upper() if v is not None else ""
        e_ipc    = _s(entry.get("ipc_section"))
        e_bns    = _s(entry.get("bns_section"))
        e_crpc   = _s(entry.get("crpc_section"))
        e_bnss   = _s(entry.get("bnss_section"))
        e_ea     = _s(entry.get("evidence_act_section"))
        e_bsa    = _s(entry.get("bsa_section"))
        e_off    = str(entry.get("offence",  entry.get("procedure", entry.get("provision", "")))).lower()
        e_what   = str(entry.get("what_changed", "")).lower()
        e_pun    = str(entry.get("punishment_note", "")).lower()
        e_reason = str(entry.get("reason", "")).lower()
        e_note   = str(entry.get("note", "")).lower()

        if ipc_q and (ipc_q in e_ipc or ipc_q in e_crpc or ipc_q in e_ea):
            return True
        if bns_q and (bns_q in e_bns or bns_q in e_bnss or bns_q in e_bsa):
            return True
        if q_lower:
            combined = f"{e_ipc} {e_bns} {e_crpc} {e_bnss} {e_ea} {e_bsa} {e_off} {e_what} {e_pun} {e_reason} {e_note}".lower()
            return q_lower in combined
        return False

    # P0-A targeted containment (2026-06). These (ipc_section, bns_section)
    # pairs were CONFIRMED INCORRECT by legal-truth audit against the NCRB
    # BNS correspondence table + official BNS Act text, and have already been
    # corrected in ipc_bns_mapping.json / build_ipc_mapping.py. Kept here ONLY
    # as a defence-in-depth tripwire: if either file is ever regenerated or
    # hand-edited and one of these exact wrong pairs reappears, the endpoint
    # must flag it as quarantined rather than silently serving it again as an
    # authoritative answer. This is intentionally a narrow, hardcoded set —
    # NOT a new global review-gating system, and does not affect any other
    # record in the dataset.
    _KNOWN_INCORRECT_IPC_BNS_PAIRS = {
        ("155", "199"), ("166A", "202"), ("168", "205"), ("255", "199"),
    }

    def _normalise(entry: dict, source: str, is_dead_law_section: bool = False) -> dict:
        """Flatten heterogeneous entries into a single shape.

        A4 — classification tag is mandatory:
          • deleted_ipc_sections entries with dead_law=True → 'dead_law'
          • deleted_ipc_sections entries without dead_law   → 'deleted'
          • new_bns_offences entries                        → 'new'
          • all others must carry their own change_type from the file;
            entries missing a tag are skipped by the caller.

        P0-A additions (additive only — no existing key removed/renamed, so
        this is backward-compatible with every current caller/consumer):
          • review_status / review_source / reviewed_at / reviewed_by /
            mapping_version — provenance fields surfaced when present on the
            record; default review_status is an honest "unreviewed_draft"
            label (NOT a block) for the ~971 other draft records in the file.
          • new_subsection — carries bns_subsection/bnss_subsection/
            bsa_subsection when a correction needs it (e.g. IPC 155 → BNS
            193(2)).
          • Targeted quarantine: if this exact (old_sec, new_sec) pair is a
            KNOWN-INCORRECT pair (see _KNOWN_INCORRECT_IPC_BNS_PAIRS above),
            change_type is forced to 'quarantined_incorrect' and what_changed
            is replaced with an honest notice — this can only ever affect
            those specific wrong pairs, never any other record.
        """
        old_sec = (
            entry.get("ipc_section")  or
            entry.get("crpc_section") or
            entry.get("evidence_act_section") or ""
        )
        new_sec = (
            entry.get("bns_section")  or
            entry.get("bnss_section") or
            entry.get("bsa_section")  or "—"
        )

        # Determine change_type, respecting A4 mandatory-tag rule
        if is_dead_law_section:
            ct = "dead_law" if entry.get("dead_law") else "deleted"
        else:
            ct = entry.get("change_type", "")

        what_changed = (
            entry.get("what_changed") or entry.get("note") or
            entry.get("reason") or entry.get("punishment_note") or ""
        )

        # Targeted quarantine tripwire (P0-A) — see docstring above.
        if (str(old_sec), str(new_sec)) in _KNOWN_INCORRECT_IPC_BNS_PAIRS:
            ct = "quarantined_incorrect"
            what_changed = (
                f"⚠️ This correspondence ({old_sec} → {new_sec}) was identified as "
                f"INCORRECT by legal audit and is pending a corrected, reviewed "
                f"value. Do not treat this as an authoritative answer — consult "
                f"an advocate or the official BNS text for the current position."
            )

        # ── CORRESPONDENCE GATE + STRUCTURAL WITHHOLDING (P1 correction) ────
        # Step 1: build canonical v1 record object from this entry's fields.
        #         This is passed to the gate so a v1 registry record can bind
        #         its hash to the SUBSTANTIVE content, not just the pair string.
        #         For the current REGISTRY-001 (v0 record) the canonical_record
        #         parameter is ignored and the v0 string hash is used instead.
        _offence_text   = str(entry.get("offence") or entry.get("procedure") or entry.get("provision") or "")
        _punishment_text = str(entry.get("punishment_note") or "")
        _has_both_sections = bool(str(old_sec) and str(new_sec) and str(new_sec) != "—")
        if _has_both_sections:
            try:
                from engine.approval_evidence import _CANONICAL_SCHEMA_VERSION  # noqa: F401
                _canonical_record: dict | None = {
                    "schema_version":  "v1",
                    "source_act":      "IPC",
                    "source_section":  str(old_sec).upper(),
                    "target_act":      "BNS",
                    "target_section":  str(new_sec).upper(),
                    "offence_text":    _offence_text,
                    "punishment_text": _punishment_text,
                }
            except Exception:
                _canonical_record = None
            _gate_result = _xref_check_corr(
                f"IPC:{str(old_sec).upper()}",
                f"BNS:{str(new_sec).upper()}",
                canonical_record=_canonical_record,
            )
            _corr_approval = _gate_result.reason
        else:
            _corr_approval = "no_section"

        # Step 2: assemble the full record object.
        _result = {
            "source":             source,
            "old_section":        str(old_sec),
            "new_section":        str(new_sec),
            "offence":            _offence_text,
            "change_type":        ct,
            "what_changed":       what_changed,
            "verified":           entry.get("verified", False),
            "source_link":        entry.get("source_link", ""),
            "judicial_citation":  entry.get("judicial_citation", ""),
            "dead_law":           bool(entry.get("dead_law", False)),
            "review_status":      entry.get("review_status", "unreviewed_draft"),
            "review_source":      entry.get("review_source", ""),
            "reviewed_at":        entry.get("reviewed_at"),
            "reviewed_by":        entry.get("reviewed_by"),
            "mapping_version":    entry.get("mapping_version", ""),
            "new_subsection":     entry.get("bns_subsection") or entry.get("bnss_subsection") or entry.get("bsa_subsection") or "",
            # CORRESPONDENCE GATE: checked on every request — never cached.
            # Anything other than "approved" means the claim is withheld.
            "correspondence_approval": _corr_approval,
        }

        # Step 3: STRUCTURAL WITHHOLDING — if the gate did NOT return "approved",
        # null out every field that would reveal the substantive BNS mapping.
        # A boolean flag or UI banner is INSUFFICIENT: the data must not leave
        # the server.  Callers receive old_section, source, change_type, and the
        # approval status so they can display a "pending approval" placeholder,
        # but nothing else.
        if _corr_approval != "approved":
            _WITHHELD_FIELDS = (
                "new_section",
                "offence",
                "what_changed",
                "source_link",
                "judicial_citation",
                "review_source",
                "reviewed_at",
                "reviewed_by",
                "mapping_version",
                "new_subsection",
            )
            for _f in _WITHHELD_FIELDS:
                _result[_f] = None
            # Replace review_status with an honest "withheld" label so the
            # frontend can distinguish between "approved data absent" and
            # "data present but unapproved".
            _result["review_status"] = "withheld_pending_approval"

        return _result

    seen: set[str] = set()
    # (key, label, is_dead_law_section)
    sources = [
        ("ipc_complete_mapping",               "IPC → BNS",        False),
        ("bns_to_ipc_mapping",                 "IPC → BNS",        False),
        ("reverse_lookup_ipc_to_bns",          "IPC → BNS",        False),
        ("crpc_to_bnss_mapping",               "CrPC → BNSS",      False),
        ("evidence_act_to_bsa_mapping",        "Evidence Act → BSA", False),
        ("new_bns_offences_no_ipc_equivalent", "New BNS",          False),
        ("deleted_ipc_sections",               "Deleted IPC",      True),
    ]
    for key, src_label, is_dead_section in sources:
        for entry in data.get(key, []):
            if _matches(entry):
                norm = _normalise(entry, src_label, is_dead_section)
                # A4: skip entries that still have no classification tag
                if not norm.get("change_type"):
                    continue
                uid  = f"{norm['old_section']}|{norm['new_section']}"
                if uid not in seen:
                    seen.add(uid)
                    results.append(norm)
                if len(results) >= 25:
                    break
        if len(results) >= 25:
            break

    return results


@api.get("/advocate/cross-reference")
async def cross_reference(
    q:   str = Query("", description="Keyword search"),
    ipc: str = Query("", description="IPC/CrPC/Evidence Act section number"),
    bns: str = Query("", description="BNS/BNSS/BSA section number"),
):
    """
    IPC → BNS cross-reference with Citation Guard.

    Citation Guard guarantees:
    1. ONLY the ipc_bns_mapping.json is the source of truth — no AI inference.
    2. Every result carries its verified/unverified status from the file.
    3. If a section is not in the file the response signals that explicitly;
       the caller MUST show the 'not in database' message — never infer a mapping.
    4. Results where bns_section is null are labelled 'deleted' — callers must
       NOT substitute a guessed BNS number.
    """
    if not q.strip() and not ipc.strip() and not bns.strip():
        raise HTTPException(400, "Supply at least one of: q, ipc, bns")

    data    = _load_xref()
    results = _xref_search(q=q, ipc=ipc, bns=bns)

    # ── Citation Guard metadata ───────────────────────────────────────────────
    is_section_search = bool(ipc.strip() or bns.strip())
    unverified_count  = sum(1 for r in results if not r.get("verified"))

    citation_guard = {
        # Always true — this endpoint never calls an LLM or infers missing data
        "database_only":      True,
        # False when a section-mode search returns 0 results → caller shows wall
        "query_found_in_db":  len(results) > 0,
        # True only when EVERY returned entry is marked verified in the file
        "all_entries_verified": unverified_count == 0 and len(results) > 0,
        "unverified_count":   unverified_count,
        "total_in_db":        sum(
            len(data.get(k, [])) for k in (
                "ipc_complete_mapping",
                "bns_to_ipc_mapping", "reverse_lookup_ipc_to_bns",
                "crpc_to_bnss_mapping", "evidence_act_to_bsa_mapping",
                "new_bns_offences_no_ipc_equivalent", "deleted_ipc_sections",
            )
        ),
        # Surface the audit trail for the caller
        "bare_act_audit_count": len(data.get("metadata", {}).get("bare_act_audit", [])),
        # What to show in the UI when query_found_in_db is False
        "not_in_db_message": (
            "Section mapping not yet confirmed in our database. "
            "Please consult a manual or the bare Act directly."
            if is_section_search and len(results) == 0 else ""
        ),
    }

    return {
        "results":            results,
        "total":              len(results),
        "citation_guard":     citation_guard,
        "which_code_applies": data.get("which_code_applies_rule", {}),
        "effective_date":     data.get("commencement_status", {}).get(
                                  "bns_bnss_bsa_effective_date", "2024-07-01"),
        "disclaimer":         data.get("metadata", {}).get("disclaimer", ""),
    }

# ─────────────────────────────────────────────────────────────────────────────

# ============================================================================
# ─── Voice FIR Drafting Assistant ────────────────────────────────────────────
# ============================================================================
from missing_media import create_missing_media_router
app.include_router(create_missing_media_router(db, current_user))
app.include_router(api)

# ── One-time build download endpoint ──────────────────────────────────────────
# Serves the latest web export zip so it can be downloaded directly via the
# preview URL (/api/download/build). Remove after the Scala upload is done.
from fastapi.responses import FileResponse as _FR
import os as _os
@app.get("/api/download/build")
async def download_build():
    _path = "/app/downloads/dhara_web_build_v19.zip"
    if not _os.path.exists(_path):
        raise HTTPException(status_code=404, detail="Build not found")
    return _FR(_path, media_type="application/zip", filename="dhara_web_build_v19.zip")
# ──────────────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_credentials=False,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def _startup():
    # Google Places (New) pilot — one-line startup confirmation so flipping
    # FEATURE_GOOGLE_PLACES_V1 + GOOGLE_PLACES_SERVER_KEY on redeploy is
    # immediately visible in server logs, with no separate script needed.
    # Never logs the key value itself — presence only.
    try:
        from api.geo_router import FEATURE_GOOGLE_PLACES_V1 as _places_flag, GOOGLE_PLACES_SERVER_KEY as _places_key
        if _places_flag and _places_key:
            logger.info("[startup] Google Places (New) pilot: ACTIVE — destination search and Nearest Help are live on Google")
        elif _places_flag and not _places_key:
            logger.warning("[startup] Google Places (New) pilot: FEATURE_GOOGLE_PLACES_V1=true but GOOGLE_PLACES_SERVER_KEY is unset — falling back to OSM/external-Maps, not an error")
        else:
            logger.info("[startup] Google Places (New) pilot: OFF (FEATURE_GOOGLE_PLACES_V1 not set) — using OSM/external-Maps as before")
    except Exception as e:
        logger.warning(f"[startup] Google Places pilot status check warning: {e}")

    # Object Storage (profile photos) — best-effort warm-up, never blocks startup
    try:
        from services.object_storage import init_storage as _init_object_storage
        await _init_object_storage()
    except Exception as e:
        logger.warning(f"[startup] object storage init warning: {e}")

    # Court Data Engine — 24h CNR lookup cache TTL index (idempotent)
    try:
        from services.court_data import ensure_cache_indexes as _ensure_court_cache_indexes
        await _ensure_court_cache_indexes()
    except Exception as e:
        logger.warning(f"[startup] court cache index creation warning: {e}")

    # Advocate Door — ensure indexes exist (idempotent)
    try:
        await db.advocate_profiles.create_index("user_id", unique=True)
        await db.advocate_profiles.create_index(
            "bar_council_number", unique=True,
            partialFilterExpression={"bar_council_number": {"$gt": ""}},
        )
        await db.client_intakes.create_index("intake_token", unique=True)
        await db.client_intakes.create_index("advocate_id")
        await db.verification_log.create_index("advocate_id")
    except Exception as e:
        logger.warning(f"[startup] advocate index creation warning: {e}")

    # C0 Journey Registry — ensure indexes exist (idempotent)
    try:
        await db.journeys.create_index("journey_id", unique=True)
        await db.journeys.create_index("category")
        await db.journeys.create_index("is_active")
    except Exception as e:
        logger.warning(f"[startup] journeys index creation warning: {e}")

    # Women's Shield Sprint A — ensure indexes exist (idempotent)
    try:
        await db.rights_cards.create_index([("scenario_key", 1), ("language", 1)], unique=True)
        await db.rights_cards.create_index("release_state")
        await db.service_directory.create_index("service_id", unique=True)
        await db.service_directory.create_index("categories")
        await db.service_directory.create_index("active")
        await db.women_audit.create_index("ts")
    except Exception as e:
        logger.warning(f"[startup] women shield index creation warning: {e}")

    # Evidence Vault (Sprint B5 cloud anchor/backup) — ensure indexes exist (idempotent)
    try:
        await db.evidence_anchors.create_index("receipt_id", unique=True)
        await db.evidence_anchors.create_index([("item_id", 1), ("scope", 1)])
        await db.evidence_backups.create_index([("item_id", 1), ("scope", 1)], unique=True)
        await db.evidence_anchors.create_index("user_id")
        await db.evidence_backups.create_index("user_id")
        await db.evidence_upload_sessions.create_index("user_id")
        await db.evidence_upload_sessions.create_index("session_id", unique=True)
        await db.evidence_upload_sessions.create_index("created_at")
    except Exception as e:
        logger.warning(f"[startup] evidence vault index creation warning: {e}")

    # Scan history (Documents tab persistence) — ensure indexes exist (idempotent)
    try:
        await db.scan_history.create_index("id", unique=True)
        await db.scan_history.create_index([("user_id", 1), ("created_at", -1)])
    except Exception as e:
        logger.warning(f"[startup] scan_history index creation warning: {e}")

    # B-16 login attempt protection — TTL index for lockout records (idempotent)
    try:
        await _login_guard_indexes(db)
    except Exception as e:
        logger.warning(f"[startup] login_attempts index creation warning: {e}")

    # Fire-and-forget: initialise Emergent Object Storage for evidence uploads
    asyncio.create_task(_init_fir_storage())

    # Fire-and-forget: sweep any abandoned Evidence Vault upload temp dirs
    asyncio.create_task(cleanup_stale_upload_sessions())

    # In-app issue reporter indexes + a 90-day TTL on automatic error events
    # (they hold no personal data, and nobody needs a year-old traceback).
    asyncio.create_task(_ensure_issue_indexes())

    async def _ensure_error_event_ttl():
        try:
            await db.error_events.create_index(
                "created_at", expireAfterSeconds=90 * 24 * 3600,
                name="error_events_ttl",
            )
        except Exception as e:
            logger.warning(f"[startup] error_events TTL index warning: {e}")

    asyncio.create_task(_ensure_error_event_ttl())

    # Fire-and-forget background task, NOT awaited: the corpus migration
    # (see corpus_migration.py) starts automatically the instant this
    # process boots, but must never delay the app from serving requests or
    # answering the platform's health check while it inserts tens of
    # thousands of documents on a cold/empty database. Every step logs
    # through the "[CORPUS_MIGRATION]" prefix regardless of how long it
    # takes; subsequent boots against an already-seeded database return
    # almost instantly (see run_corpus_migration's per-collection count
    # check) so this is never a startup cost after the first successful run.
    asyncio.create_task(run_corpus_migration(corpus_db, CORPUS_DB_NAME))

    # Fire-and-forget: the daily push-notification loop (re-engagement nudge
    # + dead-law bookmark sweep — see push_jobs.py). Sleeps until the next
    # scheduled run internally; costs nothing at boot.
    # QA Outbound Guard: scheduled push jobs never start unless OUTBOUND_MODE=live.
    if outbound_live():
        asyncio.create_task(run_push_jobs_loop(db, corpus_db))
    else:
        logger.warning("[push_jobs] NOT started — OUTBOUND_MODE=%s", outbound_mode())

    # F1 — Missed check-in deadline worker (scans every 2 min; 5-min grace period)
    try:
        from api.deadline_worker import start as _deadline_start
        _deadline_start(db)
    except Exception as e:
        logger.warning(f"[startup] deadline worker start warning: {e}")

    # ============================================================
    # DPDP Act Compliance: Automatic Retention Purge
    # ============================================================
    # Rationale: Data minimisation and purpose limitation under
    # the Digital Personal Data Protection Act, 2023.
    #
    # UPDATED (opt-in, soft-delete — explicit product-owner decision):
    # this job used to auto-run hard-deletes on every boot. A deployment
    # scan flagged that as risky for an Emergent-managed DB (irreversible,
    # silent, on by default). The owner's call: keep the DPDP-promised
    # retention periods, but (a) make the job opt-in — an operator must
    # explicitly turn it on — and (b) for `messages` specifically, scrub
    # content instead of physically deleting the row, so a mistake is
    # recoverable and there's still an audit trail of "this existed and
    # was purged on schedule". `usage_daily` (aggregate counts, not
    # personal content) remains a real hard-delete — low sensitivity,
    # nothing to recover.
    #
    # Gated by RETENTION_PURGE_ENABLED (default "false" — OPT-IN).
    # Set RETENTION_PURGE_ENABLED=true in the environment to turn this on.
    if os.getenv("RETENTION_PURGE_ENABLED", "false").strip().lower() in ("true", "1", "yes"):
        asyncio.create_task(run_retention_purge_loop())
    else:
        logger.warning(
            "[retention] RETENTION_PURGE_ENABLED is not set to true — the DPDP "
            "data-minimisation purge is OFF (opt-in by default). Set "
            "RETENTION_PURGE_ENABLED=true to enable it.",
        )


@app.on_event("shutdown")
async def _shutdown():
    # F1 — Deadline worker clean shutdown (cancel asyncio task before closing DB)
    try:
        from api.deadline_worker import stop as _deadline_stop
        await _deadline_stop()
    except Exception as e:
        logger.warning(f"[shutdown] deadline worker stop warning: {e}")
    # MongoDB connection lives in dependencies.py; close its client
    from dependencies import _mongo_client
    _mongo_client.close()


# ============================================================================
# Serve the Expo web build from this same service.
#
# The EAS deploy template's nginx proxies EVERY path to this FastAPI app
# (location / -> 127.0.0.1:8001) and builds no web bundle of its own, so
# without this the deployment answers "/" with API JSON and there is no
# website for a custom domain to point at. Every API route lives under the
# /api prefix and /health is registered near the top of this file, so both
# are matched before the catch-all below ever runs.
#
# Regenerate the bundle from frontend/ with:
#   npx expo export --platform web --output-dir ../backend/webdist

# ─── Static web bundle (must stay the LAST route in the file) ────────────────
# NOTE: GET endpoints for /api/user/my-data and /api/grievance are registered
# HERE (before the SPA catch-all) so they are not swallowed by /{web_path:path}.
# ============================================================================

# ── My Data Portal — GET (before the SPA catch-all) ──────────────────────────

@app.get("/api/user/my-data")
async def get_my_data_summary(user: dict = Depends(current_user)):
    """Return a structured summary of everything stored for the user.
    Used by the 'My Data' portal (Settings → My Data) — T&C v2.0 clause 10.3."""
    uid = user["id"]
    msg_count   = await db.messages.count_documents({"user_id": uid})
    fir_count   = await db.fir_sessions.count_documents({"user_id": uid})
    saved_count = await db.saved_answers.count_documents({"user_id": uid})
    consent_entry = await db.consent_log.find_one(
        {"user_id": uid}, sort=[("timestamp", -1)], projection={"_id": 0}
    )
    tickets = await db.grievance_tickets.find(
        {"user_id": uid},
        {"_id": 0, "ticket_id": 1, "category": 1, "status": 1, "created_at": 1}
    ).sort("created_at", -1).to_list(20)
    return {
        "profile": {
            "name":       user.get("name", ""),
            "email":      user.get("email", ""),
            "phone":      user.get("phone", ""),
            "created_at": user.get("created_at", ""),
            "language":   user.get("language", "en"),
            "state":      user.get("state", ""),
            "is_pro":     user.get("is_pro", False),
        },
        "consent": {
            "version":    consent_entry.get("notice_version") if consent_entry else user.get("terms_version"),
            "accepted_at": consent_entry.get("timestamp") if consent_entry else user.get("terms_accepted_at"),
            "purposes":   consent_entry.get("purposes") if consent_entry else user.get("consent_purposes"),
            "language":   consent_entry.get("language") if consent_entry else "en",
        },
        "activity": {
            "total_questions": msg_count,
            "fir_sessions":    fir_count,
            "saved_answers":   saved_count,
        },
        "grievances": tickets,
        "data_region": "India (MongoDB Atlas — ap-south-1)",
        "retention_policy": {
            "chat_history":     "2 years from last activity",
            "fir_drafts":       "Until account deletion + 30 days",
            "voice_recordings": "Not retained — deleted after transcription",
            "payment_records":  "7 years (tax law)",
            "consent_records":  "Indefinite (legal proof of lawful processing)",
            "otp_logs":         "24 hours (security hygiene)",
        },
    }


@app.get("/api/user/my-data/export")
async def export_my_data(user: dict = Depends(current_user)):
    """Full portable data export (JSON) — 'Export my data' button."""
    uid = user["id"]
    messages      = await db.messages.find({"user_id": uid, "purged": {"$ne": True}}, {"_id": 0}).sort("created_at", 1).to_list(5000)
    fir_sessions  = await db.fir_sessions.find({"user_id": uid}, {"_id": 0}).sort("updated_at", -1).to_list(200)
    saved_answers = await db.saved_answers.find({"user_id": uid}, {"_id": 0}).to_list(500)
    consent_log   = await db.consent_log.find({"user_id": uid}, {"_id": 0}).sort("timestamp", 1).to_list(100)
    grievances    = await db.grievance_tickets.find({"user_id": uid}, {"_id": 0}).sort("created_at", -1).to_list(50)
    return {
        "exported_at":    datetime.now(timezone.utc).isoformat(),
        "schema_version": "2.0",
        "profile":        {k: v for k, v in user.items() if k not in ("password_hash", "_id")},
        "consent_log":    consent_log,
        "messages":       messages,
        "fir_sessions":   fir_sessions,
        "saved_answers":  saved_answers,
        "grievances":     grievances,
    }


@app.get("/api/grievance")
async def list_grievances(user: dict = Depends(current_user)):
    """Return all grievance tickets raised by the authenticated user."""
    tickets = await db.grievance_tickets.find(
        {"user_id": user["id"]}, {"_id": 0}
    ).sort("created_at", -1).to_list(50)
    return {"tickets": tickets, "count": len(tickets)}


# ─── SPA catch-all (must stay LAST) ──────────────────────────────────────────
from fastapi.staticfiles import StaticFiles as _StaticFiles
from fastapi.responses import FileResponse as _FileResponse
from pathlib import Path as _WebPath

_WEB_DIR = _WebPath(__file__).parent / "webdist"

if _WEB_DIR.is_dir():
    _expo_assets = _WEB_DIR / "_expo"
    if _expo_assets.is_dir():
        app.mount("/_expo", _StaticFiles(directory=str(_expo_assets)), name="expo_assets")

    @app.get("/{web_path:path}", include_in_schema=False)
    async def _serve_expo_web(web_path: str):
        # Never swallow API calls — they have dedicated routes but FastAPI
        # resolves routes in registration order and this catch-all must not
        # win against any /api/* handler added after the static mount block.
        if web_path.startswith("api/"):
            from fastapi import HTTPException as _HE
            raise _HE(404, detail="Not Found")
        root = _WEB_DIR.resolve()
        candidate = (_WEB_DIR / web_path).resolve()
        if web_path and str(candidate).startswith(str(root)) and candidate.is_file():
            return _FileResponse(candidate)
        return _FileResponse(root / "index.html")



# ── P0-Fix4: Consent Gate endpoints ──────────────────────────────────────────

class ConsentLogIn(BaseModel):
    """Payload written by the consent gate. Works for both anonymous and
    authenticated callers; caller must supply one of user_id or anon_id."""
    notice_version: str
    purposes: dict          # {"core": true, "analytics": bool, "updates": bool}
    language: str
    age_confirmed_18: bool
    app_version: str = "1.0.0"
    anon_id: Optional[str] = None   # for pre-login / anonymous users


@app.post("/api/consent/log")
async def log_consent(
    body: ConsentLogIn,
    user: Optional[dict] = Depends(current_user_optional),
):
    """Append-only consent log (DPDP §6, PIPEDA Principle 3).
    Never updates or deletes rows — only inserts."""
    entry = {
        "user_id":          user["id"] if user else None,
        "anon_id":          body.anon_id if not user else None,
        "notice_version":   body.notice_version,
        "purposes":         body.purposes,
        "language":         body.language,
        "age_confirmed_18": body.age_confirmed_18,
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "app_version":      body.app_version,
    }
    await db.consent_log.insert_one(entry)

    # If authenticated, also update the user record with the latest consent info
    if user:
        await db.users.update_one(
            {"id": user["id"]},
            {"$set": {
                "terms_version":      body.notice_version,
                "terms_accepted":     True,
                "terms_accepted_at":  entry["timestamp"],
                "consent_purposes":   body.purposes,
                "age_confirmed_18":   body.age_confirmed_18,
                "consent_language":   body.language,
            }},
        )
    return {"ok": True}


class PrivacyChoicesIn(BaseModel):
    analytics: bool
    updates: bool


@app.patch("/api/user/privacy-choices")
async def update_privacy_choices(
    body: PrivacyChoicesIn,
    user: dict = Depends(current_user),
):
    """Let the user flip optional consent purposes (DPDP right to withdraw)."""
    new_purposes = {"core": True, "analytics": body.analytics, "updates": body.updates}
    now = datetime.now(timezone.utc).isoformat()
    await db.users.update_one(
        {"id": user["id"]},
        {"$set": {"consent_purposes": new_purposes, "terms_accepted_at": now}},
    )
    # Write a consent_log entry so every change is auditable
    await db.consent_log.insert_one({
        "user_id":          user["id"],
        "anon_id":          None,
        "notice_version":   user.get("terms_version", ""),
        "purposes":         new_purposes,
        "language":         user.get("consent_language", "en"),
        "age_confirmed_18": user.get("age_confirmed_18", True),
        "timestamp":        now,
        "app_version":      "1.0.0",
        "event":            "privacy_choices_update",
    })
    return {"ok": True, "consent_purposes": new_purposes}



# ═══════════════════════════════════════════════════════════════════════════════
# MY DATA PORTAL — endpoints defined before the SPA catch-all (see above)
# POST/PATCH endpoints that don't conflict with the catch-all stay here for
# structural clarity alongside the Pydantic models and grievance POST.
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# GRIEVANCE TICKETS — DPDP Ch. IV: Grievance Redressal
# ═══════════════════════════════════════════════════════════════════════════════

GRIEVANCE_CATEGORIES = {
    "data_access":     "Request to access my data",
    "data_correction": "Request to correct my data",
    "data_deletion":   "Request to delete my data",
    "objection":       "Object to data processing",
    "data_breach":     "Report a suspected data breach",
    "other":           "Other privacy concern",
}


class GrievanceIn(BaseModel):
    category: str           # one of GRIEVANCE_CATEGORIES keys
    description: str = Field(default="", max_length=2000)
    contact_email: Optional[str] = None  # alternate contact; defaults to account email


@app.post("/api/grievance")
async def submit_grievance(body: GrievanceIn, user: dict = Depends(current_user)):
    """Create a privacy grievance ticket (DPDP §13(3) / PIPEDA §11).
    Returns a reference ID the user can quote when following up."""
    if body.category not in GRIEVANCE_CATEGORIES:
        raise HTTPException(400, f"Unknown category. Valid values: {list(GRIEVANCE_CATEGORIES)}")

    import random, string as _string
    suffix = "".join(random.choices(_string.ascii_uppercase + _string.digits, k=6))
    ticket_id = f"GRV-{datetime.now(timezone.utc).strftime('%Y%m')}-{suffix}"

    doc = {
        "ticket_id":   ticket_id,
        "user_id":     user["id"],
        "user_email":  body.contact_email or user.get("email", ""),
        "category":    body.category,
        "description": body.description.strip(),
        "status":      "received",          # received → acknowledged → resolved
        "created_at":  datetime.now(timezone.utc).isoformat(),
        "updated_at":  datetime.now(timezone.utc).isoformat(),
    }
    await db.grievance_tickets.insert_one(doc)

    return {
        "ok":        True,
        "ticket_id": ticket_id,
        "category":  GRIEVANCE_CATEGORIES[body.category],
        "status":    "received",
        "message":   (
            f"Your grievance has been recorded (reference: {ticket_id}). "
            "We will acknowledge it within 48 hours and aim to resolve it within 30 days "
            "as required under the DPDP Act 2023."
        ),
    }


# list_grievances GET is defined before the SPA catch-all (see ~line 3762).


# POST /api/user/data-export — stub endpoint (DPDP compliance notice screen)
# Logs the request and confirms via API response. Actual export is sent by email
# within 24 hours. No file generation needed in this sprint.
@app.post("/api/user/data-export")
async def request_data_export(user: dict = Depends(current_user)):
    """Log a data export request and return a confirmation.
    Fulfils: DHARA Data & Privacy screen → 'Download my data' card.
    Actual data package is prepared and emailed within 24 hours."""
    await db.data_export_requests.insert_one({
        "user_id":      user["id"],
        "email":        user.get("email", ""),
        "requested_at": datetime.now(timezone.utc).isoformat(),
        "status":       "pending",
    })
    return {
        "ok": True,
        "message": "We'll email your data export within 24 hours.",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# RETENTION PURGE JOB — DPDP §8(3): Data minimisation
# consent_log is EXEMPT — retained indefinitely per T&C v2.0 clause 10.5
#
# ⚠️ THIS IS A DELIBERATE COMPLIANCE CONTROL, NOT A BUG OR HOUSEKEEPING TASK.
# Do not remove, soft-delete-ify, or disable this job without first updating
# privacy.html's retention section and notifying the Data Protection Board if
# required. Rationale (confirmed with product owner, 28 Sep 2026):
#   - DPDP Act data-minimisation and purpose-limitation are obligations, not
#     options — a citizen asking about their FIR/rights has no reasonable
#     expectation DHARA retains that query 2 years later.
#   - A scheduled purge on a documented, published retention schedule is
#     exactly what the Data Protection Board expects to see as evidence of
#     compliance if ever audited.
#   - Soft-delete would create a new category of data that exists but is
#     inaccessible to users — a disclosure/erasure liability with no user
#     benefit — so it is deliberately NOT used here.
#   - Disabling the purge means indefinite retention, which directly
#     contradicts data minimisation and would require a privacy-policy
#     update before it could ship.
# ═══════════════════════════════════════════════════════════════════════════════

async def run_retention_purge_loop():
    """Daily background job that purges records past their retention period.
    OFF by default (RETENTION_PURGE_ENABLED, opt-in) — see the startup-event
    comment block above for why this changed from "auto-on, hard-delete".

    Retention periods (T&C v2.0, clause 10.5 — must match /api/user/my-data
    "retention_policy" exactly; both were reconciled together on 4 Oct 2026):
    - messages                 : 730 days (2 years) from created_at —
                                  SOFT-DELETE: content scrubbed in place,
                                  `purged: true` set, row kept as a tombstone.
                                  Every read path that surfaces message
                                  content to a user (chat history, data
                                  export) excludes purged=true rows, so the
                                  user-facing effect is identical to a real
                                  delete.
    - fir_sessions              : NOT age-purged while the account is active.
                                  Only purged 30 days after the OWNER'S account
                                  was deleted (account_deleted_at stamped by
                                  account_deletion.hard_delete_user). Matches
                                  "fir_drafts: until account deletion + 30 days".
    - usage_daily              : 90 days (roll-up window) — real hard-delete;
                                  aggregate counters only, no personal content.
    - password_resets / otps   : 24 hours (security hygiene) — real hard-delete.
    - consent_log              : EXEMPT — never purged (legal proof)
    - grievance_tickets        : EXEMPT — purge only by explicit deletion request
    """
    while True:
        try:
            now = datetime.now(timezone.utc)

            # ── messages: 730 days — SOFT-DELETE (scrub content, keep tombstone) ──
            cutoff_2yr = (now - timedelta(days=730)).isoformat()
            r = await db.messages.update_many(
                {"created_at": {"$lt": cutoff_2yr}, "purged": {"$ne": True}},
                {"$set": {"content": "[purged — retention period expired]", "citations": [], "purged": True, "purged_at": now.isoformat()}},
            )
            if r.modified_count:
                logger.info(f"[retention] messages soft-purged (content scrubbed): {r.modified_count}")

            # ── fir_sessions: 30 days AFTER the owning account was deleted ──
            # (NOT a blanket age purge — an active user's FIR draft must
            # survive indefinitely; only tombstoned rows, stamped by
            # account_deletion.hard_delete_user with account_deleted_at,
            # are eligible, and only once 30 days have passed since then.)
            cutoff_30d = (now - timedelta(days=30)).isoformat()
            r = await db.fir_sessions.delete_many({"account_deleted_at": {"$lt": cutoff_30d}})
            if r.deleted_count:
                logger.info(f"[retention] fir_sessions purged (post account-deletion grace period): {r.deleted_count}")

            # ── usage_daily: 90 days (keyed on 'day' string YYYY-MM-DD) ────
            day_90 = (now - timedelta(days=90)).strftime("%Y-%m-%d")
            r = await db.usage_daily.delete_many({"day": {"$lt": day_90}})
            if r.deleted_count:
                logger.info(f"[retention] usage_daily purged: {r.deleted_count}")

            # ── OTPs / password resets: 24 hours ───────────────────────────
            cutoff_24h = (now - timedelta(hours=24)).isoformat()
            for col in ("password_resets", "account_deletion_otps"):
                r = await getattr(db, col).delete_many({"created_at": {"$lt": cutoff_24h}})
                if r.deleted_count:
                    logger.info(f"[retention] {col} purged: {r.deleted_count}")

            logger.info("[retention] daily purge cycle complete")
        except Exception:
            logger.exception("[retention] purge job error — will retry in 1 h")
            await asyncio.sleep(3600)
            continue

        # Sleep 24 h before next cycle
        await asyncio.sleep(86400)
