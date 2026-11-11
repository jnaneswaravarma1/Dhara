"""
Visitor Mode — Translation endpoint.
POST /api/visitor/translate  {text, source_lang, target_lang} -> {translated_text}
Uses Claude Haiku 4.5 via Emergent LLM key — fast and cheap.
"""
import os
import uuid
import logging
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from services.ai_gateway import collect_stream, VISITOR_TRANSLATE_SPEC
from dependencies import current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/visitor", tags=["visitor"])

EMERGENT_KEY = os.environ.get("EMERGENT_LLM_KEY", "")

# ISO 639-1 → language name for the system prompt
_LANG_NAMES: dict[str, str] = {
    "en": "English",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "pt": "Portuguese",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese (Simplified)",
    "ar": "Arabic",
    "ru": "Russian",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "bn": "Bengali",
    "kn": "Kannada",
    "gu": "Gujarati",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "as": "Assamese",
    "or": "Odia",
    "ur": "Urdu",
    "kok": "Konkani",
}


class TranslateIn(BaseModel):
    text: str
    source_lang: str  # ISO 639-1 e.g. "en"
    target_lang: str  # ISO 639-1 e.g. "hi"


@router.post("/translate")
async def translate(
    body: TranslateIn,
    user: dict = Depends(current_user),
):
    """
    Translate `text` from `source_lang` to `target_lang`.
    Used by the Two-Way Interpreter in Visitor Mode.
    """
    if not body.text or not body.text.strip():
        raise HTTPException(400, "text is required")
    if not EMERGENT_KEY:
        raise HTTPException(503, "LLM key not configured")

    src_name = _LANG_NAMES.get(body.source_lang, body.source_lang)
    tgt_name = _LANG_NAMES.get(body.target_lang, body.target_lang)

    system = (
        f"You are a professional interpreter. "
        f"Translate the following {src_name} text to {tgt_name}. "
        f"Output ONLY the translated text—no explanations, no notes, no quotes."
        # Speak For Me fix: the model sometimes returned a half-translated line
        # plus a second version, or kept source-language words.
        f" Give exactly ONE translation, written entirely in natural {tgt_name}"
        f" with no {src_name} words left in it. If both languages use the same"
        f" script, still rewrite every word (pronouns, verbs, endings) in"
        f" {tgt_name} grammar — do not copy {src_name} words across."
    )

    try:
        # Routed through the AI gateway (model from VISITOR_TRANSLATE_SPEC);
        # streamed tokens are collected into a single string.
        translated = await collect_stream(
            str(uuid.uuid4()),
            system,
            body.text,
            spec=VISITOR_TRANSLATE_SPEC,
            caller="visitor_translate",
        )
        translated = translated.strip()
        if not translated:
            raise ValueError("empty translation")
        return {"translated_text": translated}
    except Exception as e:
        logger.exception("translate failed: %s", e)
        raise HTTPException(500, f"Translation failed: {e}")
