"""Central configuration — loaded once, imported everywhere.

All env-var reads live here. Feature modules import what they need.
Model identifiers are environment-driven; no hard-coded model strings
outside this file.
"""
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).parent.parent
load_dotenv(ROOT_DIR / ".env")

# ── Database ───────────────────────────────────────────────────────────────────
MONGO_URL         = os.environ["MONGO_URL"]
DB_NAME           = os.environ["DB_NAME"]
CORPUS_DB_NAME    = os.environ.get("CORPUS_DB_NAME", "bns-know-your-rights-gandhikar_db")

# ── LLM ───────────────────────────────────────────────────────────────────────
EMERGENT_LLM_KEY  = os.environ["EMERGENT_LLM_KEY"]

# Server-side model config — clients never choose these
SERVER_CHAT_PROVIDER      = os.getenv("CHAT_PROVIDER",         "anthropic")
SERVER_CHAT_MODEL         = os.getenv("CHAT_MODEL",            "claude-sonnet-4-5-20250929")
SERVER_CHAT_FALLBACK_MODEL = os.getenv("CHAT_FALLBACK_MODEL",  "claude-haiku-4-5")

# Secondary task-specific models (translation, document-scan vision). Kept
# separate from the main chat spec above since they run cheaper, task-tuned
# models — but still centralised here so no router hard-codes a model string.
TRANSLATE_PROVIDER = os.getenv("TRANSLATE_PROVIDER", "openai")
TRANSLATE_MODEL    = os.getenv("TRANSLATE_MODEL",    "gpt-4o-mini")
SCAN_PROVIDER       = os.getenv("SCAN_PROVIDER",      "openai")
SCAN_MODEL          = os.getenv("SCAN_MODEL",         "gpt-4o-mini")

# Task-specific text models that used to be hard-coded at their call sites.
# Defaults are EXACTLY the strings those call sites used, so behaviour is
# unchanged until an env var overrides one. All run on TASK_PROVIDER.
TASK_PROVIDER             = os.getenv("TASK_PROVIDER",             "anthropic")
QUERY_PARSE_MODEL         = os.getenv("QUERY_PARSE_MODEL",         "claude-haiku-4-5")
RETRIEVAL_KEYWORD_MODEL   = os.getenv("RETRIEVAL_KEYWORD_MODEL",   "claude-haiku-4-5")
FIR_EXTRACT_MODEL         = os.getenv("FIR_EXTRACT_MODEL",         "claude-haiku-4-5")
FIR_DRAFT_MODEL           = os.getenv("FIR_DRAFT_MODEL",           "claude-sonnet-4-6")
FOLLOWUP_MODEL            = os.getenv("FOLLOWUP_MODEL",            "claude-sonnet-4-5-20250929")
VISITOR_TRANSLATE_MODEL   = os.getenv("VISITOR_TRANSLATE_MODEL",   "claude-haiku-4-5-20251001")
INTAKE_SUMMARY_MODEL      = os.getenv("INTAKE_SUMMARY_MODEL",      "claude-sonnet-4-5-20250929")

# ── Auth / JWT ─────────────────────────────────────────────────────────────────
JWT_SECRET   = os.environ["JWT_SECRET"]
JWT_ALG      = "HS256"
JWT_EXP_DAYS = 30

# ── Billing ────────────────────────────────────────────────────────────────────
STRIPE_API_KEY           = os.environ.get("STRIPE_API_KEY",           "")
STRIPE_WEBHOOK_SECRET    = os.environ.get("STRIPE_WEBHOOK_SECRET",    "")
RAZORPAY_KEY_ID          = os.environ.get("RAZORPAY_KEY_ID",          "")
RAZORPAY_KEY_SECRET      = os.environ.get("RAZORPAY_KEY_SECRET",      "")
RAZORPAY_WEBHOOK_SECRET  = os.environ.get("RAZORPAY_WEBHOOK_SECRET",  "")
RAZORPAY_ME_HANDLE       = os.environ.get("RAZORPAY_ME_HANDLE",       "calviltech")
RAZORPAY_ME_URL          = os.environ.get("RAZORPAY_ME_URL",          "https://razorpay.me/@calviltech")

PRO_PRICE_INR       = int(os.environ.get("PRO_PRICE_INR",   "9900"))
PRO_PRICE_LABEL     = os.environ.get("PRO_PRICE_LABEL",     "₹99")
PRO_PRICE_USD       = int(os.environ.get("PRO_PRICE_USD",   "500"))
PRO_PRICE_USD_LABEL = os.environ.get("PRO_PRICE_USD_LABEL", "$5")
PRO_FREE_SAMPLES    = int(os.environ.get("PRO_FREE_SAMPLES","5"))
DRAFTS_FREE         = int(os.environ.get("DRAFTS_FREE",     "1"))

# ── Rate limits ────────────────────────────────────────────────────────────────
FREE_DAILY_QUERIES  = int(os.environ.get("FREE_DAILY_QUERIES",  "30"))
APP_DAILY_LLM_CALLS = int(os.environ.get("APP_DAILY_LLM_CALLS", "3000"))

# ── External APIs ──────────────────────────────────────────────────────────────
ECOURTS_TOKEN = os.getenv("ECOURTS_API_TOKEN", "")
ECOURTS_BASE  = os.getenv("ECOURTS_API_BASE",  "https://webapi.ecourtsindia.com")

# 24h cache on successful CNR lookups so a repeated hearing-date check does
# not re-hit the eCourts partner API. See services/court_data.py.
COURT_CNR_CACHE_TTL_S = int(os.getenv("COURT_CNR_CACHE_TTL_S", "86400"))

# Ops alert channels for the eCourts circuit breaker (see services/alerts.py).
# Both optional — leave blank to disable. Fill in later without a redeploy
# by editing backend/.env.
OPS_ALERT_EMAIL             = os.getenv("OPS_ALERT_EMAIL", "").strip()
OPS_ALERT_SLACK_WEBHOOK_URL = os.getenv("OPS_ALERT_SLACK_WEBHOOK_URL", "").strip()

# ── Language registry (22 official + English) ─────────────────────────────────
LANGUAGES = [
    {"code": "en",  "name": "English",    "native": "English",         "tts": "en-IN"},
    {"code": "hi",  "name": "Hindi",      "native": "हिन्दी",           "tts": "hi-IN"},
    {"code": "bn",  "name": "Bengali",    "native": "বাংলা",            "tts": "bn-IN"},
    {"code": "ta",  "name": "Tamil",      "native": "தமிழ்",            "tts": "ta-IN"},
    {"code": "te",  "name": "Telugu",     "native": "తెలుగు",           "tts": "te-IN"},
    {"code": "mr",  "name": "Marathi",    "native": "मराठी",            "tts": "mr-IN"},
    {"code": "gu",  "name": "Gujarati",   "native": "ગુજરાતી",          "tts": "gu-IN"},
    {"code": "kn",  "name": "Kannada",    "native": "ಕನ್ನಡ",            "tts": "kn-IN"},
    {"code": "ml",  "name": "Malayalam",  "native": "മലയാളം",           "tts": "ml-IN"},
    {"code": "pa",  "name": "Punjabi",    "native": "ਪੰਜਾਬੀ",           "tts": "pa-IN"},
    {"code": "or",  "name": "Odia",       "native": "ଓଡ଼ିଆ",            "tts": "or-IN"},
    {"code": "as",  "name": "Assamese",   "native": "অসমীয়া",           "tts": "as-IN"},
    {"code": "ur",  "name": "Urdu",       "native": "اردو",             "tts": "ur-IN"},
    {"code": "sd",  "name": "Sindhi",     "native": "سنڌي",             "tts": "sd-IN"},
    {"code": "ks",  "name": "Kashmiri",   "native": "कॉशुर",            "tts": "ks-IN"},
    {"code": "ne",  "name": "Nepali",     "native": "नेपाली",           "tts": "ne-NP"},
    {"code": "sa",  "name": "Sanskrit",   "native": "संस्कृतम्",         "tts": "sa-IN"},
    {"code": "kok", "name": "Konkani",    "native": "कोंकणी",           "tts": "kok-IN"},
    {"code": "mai", "name": "Maithili",   "native": "मैथिली",           "tts": "mai-IN"},
    {"code": "mni", "name": "Manipuri",   "native": "মৈতৈলোন্",         "tts": "mni-IN"},
    {"code": "sat", "name": "Santali",    "native": "ᱥᱟᱱᱛᱟᱲᱤ",          "tts": "sat-IN"},
    {"code": "doi", "name": "Dogri",      "native": "डोगरी",            "tts": "doi-IN"},
    {"code": "brx", "name": "Bodo",       "native": "बर'",              "tts": "brx-IN"},
]

# ── Feature Flags (server-controlled, read by /health/ready) ──────────────────
# Set to "false" / "0" to disable a subsystem without a redeploy.
FEATURE_CHAT   = os.getenv("FEATURE_CHAT",   "true").lower() not in ("false", "0", "off")
FEATURE_VOICE  = os.getenv("FEATURE_VOICE",  "true").lower() not in ("false", "0", "off")
FEATURE_FIR    = os.getenv("FEATURE_FIR",    "true").lower() not in ("false", "0", "off")
FEATURE_COURT  = os.getenv("FEATURE_COURT",  "true").lower() not in ("false", "0", "off")
FEATURE_VOTER  = os.getenv("FEATURE_VOTER",  "true").lower() not in ("false", "0", "off")
FEATURE_MISSING_PERSON = os.getenv("FEATURE_MISSING_PERSON", "true").lower() not in ("false", "0", "off")
FEATURE_BILLING = os.getenv("FEATURE_BILLING", "true").lower() not in ("false", "0", "off")

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
