"""FIR Voice Interview Engine, evidence management, and Voter Roll PDF routes.

Extracted from server.py (Phase 1 Modularization).
Depends on: dependencies.py, config/settings.py
"""
import io
import uuid
import datetime
from datetime import datetime as _dt, timezone
from typing import Optional, List
from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Request
from fastapi.responses import StreamingResponse, Response as _Resp
from pydantic import BaseModel

from services.ai_gateway import send_once, FOLLOWUP_SPEC
from dependencies import db, corpus_db, current_user, current_user_optional, logger
from config.settings import EMERGENT_LLM_KEY
from fir_engine import (
    create_session, process_turn,
    STAGE_COMPLETED, STAGE_SAFETY_GATE,
)
from fir_storage import init_storage, upload_evidence, download_evidence

router = APIRouter()


async def _get_owned_session(session_id: str, user: Optional[dict], projection: Optional[dict] = None) -> dict:
    """Fetch a FIR session and enforce ownership (SEC-001 fix, extending the
    existing P0-Fix1 pattern from /fir/session/{id}/draft repo-wide).

    Anonymous sessions (user_id starts with 'anon-', or empty) are owned by
    whoever holds the session_id UUID — that UUID is never exposed except
    through the interview flow itself, so knowledge of it is treated as
    sufficient proof of ownership (unchanged behaviour for anonymous users,
    who must not be blocked by a login wall while reporting an incident).

    Sessions created by a real account require the caller to be
    authenticated AND to match the recorded owner — this is the part that
    was previously missing on most routes below.
    """
    doc = await db.fir_sessions.find_one({"session_id": session_id}, projection or {"_id": 0})
    if not doc:
        raise HTTPException(404, "Session not found")
    owner_id: str = doc.get("user_id") or ""
    is_anon = owner_id.startswith("anon-") or owner_id == ""
    if not is_anon:
        if user is None:
            raise HTTPException(401, "Authentication required")
        if owner_id != user["id"]:
            raise HTTPException(403, "Access denied")
    return doc


class FirDraftIn(BaseModel):
    draft_id: Optional[str] = None          # None → create new
    user_id: str
    language: str = "en"
    answers: Optional[dict] = None          # {q_id: answer_text}
    status: str = "in_progress"             # in_progress | completed

class FirSessionIn(BaseModel):
    user_id: str
    language: str = "en"
    session_location_start: Optional[dict] = None   # {"lat": ..., "lng": ..., "address": "..."}
    incident_type: Optional[str] = None             # pre-selected module: "cybercrime" | "domestic_violence" | "theft" | "posh" | "consumer_fraud" | "other"


class FirTurnIn(BaseModel):
    user_message: Optional[str] = None
    gps: Optional[dict] = None              # {"lat": ..., "lng": ...}  for GPS probe
    action: Optional[str] = None            # "skip" | "upload_done" | "confirm" | "not_safe"
    language: Optional[str] = None          # propagate selected language on every turn


class FirGpsLogIn(BaseModel):
    lat: float
    lng: float
    address: Optional[str] = None
    log_type: str = "start"                 # "start" | "end"


class ChatFollowupIn(BaseModel):
    message: str          # original user question
    answer: str           # AI answer (first 500 chars used)
    language: str = "en"
    language_name: str = "English"


@router.post("/fir/draft")
async def fir_upsert_draft(body: FirDraftIn):
    """Create or update a FIR draft (upsert by draft_id).
    Intentionally NOT behind current_user — the Voice FIR Drafting
    Assistant is anonymous-first (a citizen must not be blocked by a
    login wall while reporting an incident). `user_id` may be a real
    logged-in user's id, or a locally-generated anonymous id supplied
    by the frontend for guests."""
    now = _dt.now(timezone.utc).isoformat()
    if body.draft_id:
        doc = await db.fir_drafts.find_one({"id": body.draft_id, "user_id": body.user_id})
        if doc:
            await db.fir_drafts.update_one(
                {"id": body.draft_id},
                {"$set": {"answers": body.answers or {}, "status": body.status,
                          "language": body.language, "updated_at": now}}
            )
            return {"draft_id": body.draft_id, "status": body.status}
    # Create new
    draft_id = str(uuid.uuid4())
    await db.fir_drafts.insert_one({
        "id": draft_id, "user_id": body.user_id, "language": body.language,
        "answers": body.answers or {}, "status": body.status,
        "draft_text": "", "document_checklist": [], "classification": {},
        "police_station": {}, "safety_flags": [], "created_at": now, "updated_at": now,
    })
    return {"draft_id": draft_id, "status": body.status}


@router.get("/fir/drafts/{user_id}")
async def fir_list_drafts(user_id: str, user: Optional[dict] = Depends(current_user_optional)):
    """List FIR drafts for a user_id. SEC-001 fix: a real user_id requires
    the caller to be authenticated AND match it; an anon-id is owned by
    whoever holds that exact (unguessable, UUID-based) id string."""
    is_anon = user_id.startswith("anon-") or user_id == ""
    if not is_anon:
        if user is None:
            raise HTTPException(401, "Authentication required")
        if user_id != user["id"]:
            raise HTTPException(403, "Access denied")
    docs = await db.fir_drafts.find(
        {"user_id": user_id}, {"_id": 0}
    ).sort("updated_at", -1).to_list(50)
    return docs


@router.get("/fir/draft/{draft_id}")
async def fir_get_draft(draft_id: str, user: Optional[dict] = Depends(current_user_optional)):
    doc = await db.fir_drafts.find_one({"id": draft_id}, {"_id": 0})
    if not doc:
        raise HTTPException(404, "Draft not found")
    owner_id: str = doc.get("user_id") or ""
    is_anon = owner_id.startswith("anon-") or owner_id == ""
    if not is_anon:
        if user is None:
            raise HTTPException(401, "Authentication required")
        if owner_id != user["id"]:
            raise HTTPException(403, "Access denied")
    return doc


@router.post("/fir/classify")
async def fir_classify_gone():
    raise HTTPException(410, "Removed. Use POST /api/fir/session")


@router.post("/fir/generate")
async def fir_generate_gone():
    raise HTTPException(410, "Removed. Use POST /api/fir/session/{id}/turn")


@router.post("/fir/event")
async def fir_event_gone():
    raise HTTPException(410, "Removed.")


@router.post("/chat/followup_fir_stub")  # moved to chat_router.py; stub to prevent 404 on stale clients
async def chat_followup(body: ChatFollowupIn, user: dict = Depends(current_user)):
    """
    Generate 2-3 short follow-up question chips after an AI chat answer.
    These appear as tappable chips below the answer to guide the next question.
    Returns { questions: ["q1", "q2", "q3"] } or { questions: [] }.
    """
    if not body.answer or len(body.answer.strip()) < 20:
        return {"questions": []}

    lang_instruction = f"in {body.language_name}" if body.language_name.lower() != "english" else "in English"

    prompt = f"""A citizen asked a legal question.

Question: "{body.message[:200]}"
Answer summary: "{body.answer[:400]}"

Generate exactly 2-3 SHORT natural follow-up questions the citizen might ask next, {lang_instruction}.

Rules:
- Each question must be under 9 words
- Must be directly relevant to the topic
- Phrased as a brief question (not statements)
- No duplicates of the original question

Reply ONLY with a valid JSON array of strings, nothing else:
["question 1", "question 2", "question 3"]"""

    try:
        import json as _json
        result = await send_once(
            f"chat-followup-{id(body)}",
            "Generate short follow-up questions as a JSON array only.",
            prompt,
            spec=FOLLOWUP_SPEC,
            caller="chat_followup",
        )
        raw = (result or "[]").strip()

        # Find JSON array in response
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


@router.post("/fir/followup")
async def fir_followup_gone():
    raise HTTPException(410, "Removed. Use POST /api/fir/session/{id}/turn")


# ─── NEW: FIR Session Interview Engine ─────────────────────────────────────────


@router.post("/fir/session")
async def fir_create_session(body: FirSessionIn, user: Optional[dict] = Depends(current_user_optional)):
    """Create a fresh, blank FIR session (Safety Gate stage).
    SEC-001 hardening: a logged-in caller cannot attribute a new session to
    someone else's account — their own id always wins over the body."""
    owner_id = user["id"] if user else body.user_id
    return await create_session(
        db,
        user_id=owner_id,
        language=body.language,
        session_location_start=body.session_location_start,
        incident_type=body.incident_type,
    )


@router.post("/fir/session/{session_id}/turn")
async def fir_session_turn(session_id: str, body: FirTurnIn, user: Optional[dict] = Depends(current_user_optional)):
    """Process one conversation turn in the FIR interview engine."""
    await _get_owned_session(session_id, user, {"_id": 0, "user_id": 1})
    # Propagate selectedLanguage into the session on every turn so language changes
    # mid-session are honoured immediately (language sync fix).
    if body.language:
        await db.fir_sessions.update_one(
            {"session_id": session_id},
            {"$set": {"language": body.language, "updated_at": _dt.now(timezone.utc).isoformat()}},
        )
    return await process_turn(
        db=db,
        corpus_db=corpus_db,
        session_id=session_id,
        user_message=body.user_message,
        gps=body.gps,
        action=body.action,
        llm_key=EMERGENT_LLM_KEY,
    )


@router.post("/fir/session/{session_id}/gps")
async def fir_log_gps(session_id: str, body: FirGpsLogIn, user: Optional[dict] = Depends(current_user_optional)):
    """Silently log GPS at session start or end."""
    await _get_owned_session(session_id, user, {"_id": 0, "user_id": 1})
    gps_data = {"lat": body.lat, "lng": body.lng, "address": body.address or "",
                "timestamp": _dt.now(timezone.utc).isoformat()}
    field = "session_location_start" if body.log_type == "start" else "session_location_end"
    await db.fir_sessions.update_one(
        {"session_id": session_id},
        {"$set": {field: gps_data, "updated_at": _dt.now(timezone.utc).isoformat()}},
    )
    return {"ok": True}


@router.post("/fir/session/{session_id}/evidence")
async def fir_upload_evidence(session_id: str, file: UploadFile = File(...), user: Optional[dict] = Depends(current_user_optional)):
    """Upload one evidence file (max 20 MB). Stored via Emergent Object Storage."""
    MAX_SIZE = 20 * 1024 * 1024
    data = await file.read()
    if len(data) > MAX_SIZE:
        raise HTTPException(413, "File too large — maximum 20 MB per file")

    session = await _get_owned_session(session_id, user)
    if len(session.get("evidence_files", [])) >= 10:
        raise HTTPException(400, "Maximum 10 files per session")

    content_type = file.content_type or "application/octet-stream"
    filename = file.filename or "evidence"

    meta = await upload_evidence(
        user_id=session.get("user_id", "anon"),
        session_id=session_id,
        filename=filename,
        data=data,
        content_type=content_type,
    )
    await db.fir_sessions.update_one(
        {"session_id": session_id},
        {"$push": {"evidence_files": meta},
         "$set": {"updated_at": _dt.now(timezone.utc).isoformat()}},
    )
    return {"ok": True, "file": meta}


@router.get("/fir/session/{session_id}/evidence/{file_id}")
async def fir_download_evidence(session_id: str, file_id: str, user: Optional[dict] = Depends(current_user_optional)):
    """Download an evidence file by file_id."""
    from fastapi.responses import Response
    session = await _get_owned_session(session_id, user)
    ev = next((e for e in session.get("evidence_files", []) if e.get("file_id") == file_id), None)
    if not ev:
        raise HTTPException(404, "File not found")
    data, content_type = await download_evidence(ev["storage_path"])
    return Response(content=data, media_type=content_type)


@router.get("/fir/sessions/{user_id}")
async def fir_list_sessions(user_id: str, user: Optional[dict] = Depends(current_user_optional)):
    """List all FIR sessions for a user (for pause/resume). SEC-001 fix:
    a real user_id requires the caller to be authenticated AND match it."""
    is_anon = user_id.startswith("anon-") or user_id == ""
    if not is_anon:
        if user is None:
            raise HTTPException(401, "Authentication required")
        if user_id != user["id"]:
            raise HTTPException(403, "Access denied")
    docs = await db.fir_sessions.find(
        {"user_id": user_id}, {"_id": 0, "narrative_turns": 0}
    ).sort("updated_at", -1).to_list(20)
    return docs


@router.get("/fir/session/{session_id}")
async def fir_get_session(session_id: str, user: Optional[dict] = Depends(current_user_optional)):
    """Get a FIR session by session_id."""
    return await _get_owned_session(session_id, user)


# ── P0-Fix1: Secure draft retrieval — no PII in URL ───────────────────────────
@router.get("/fir/session/{session_id}/draft")
async def fir_get_session_draft(
    session_id: str,
    request: Request,
    user: Optional[dict] = Depends(current_user_optional),
):
    """Return the generated FIR draft text for a session.

    Security rules (P0-Fix1):
    • Logged-in user: must own the session (session.user_id == user.id) → 403 otherwise.
    • Anonymous session (user_id starts with 'anon-'): knowledge of the UUID
      session_id is treated as sufficient proof of ownership (the UUID is never
      in the URL, only obtained from the interview flow itself).
    • No generated draft yet → 404 (interview not complete).
    """
    doc = await db.fir_sessions.find_one(
        {"session_id": session_id},
        {"_id": 0, "draft_text": 1, "draft": 1, "user_id": 1},
    )
    if not doc:
        raise HTTPException(404, "Session not found")

    owner_id: str = doc.get("user_id") or ""
    is_anon = owner_id.startswith("anon-") or owner_id == ""

    if not is_anon:
        # Require authenticated caller who owns this session
        if user is None:
            raise HTTPException(401, "Authentication required")
        if owner_id != user["id"]:
            raise HTTPException(403, "Access denied")

    # The FIR engine persists completed letters in `draft`; retain compatibility
    # with earlier documents that used `draft_text`. Ownership checks stay above.
    draft = doc.get("draft_text") or doc.get("draft") or ""
    if not draft:
        raise HTTPException(404, "Draft not ready yet")

    return {"session_id": session_id, "draft_text": draft}


# ── v3.3: Pause session ───────────────────────────────────────────────────────
@router.post("/fir/session/{session_id}/pause")
async def fir_pause_session(session_id: str, user: Optional[dict] = Depends(current_user_optional)):
    """Mark session as paused (Save & Continue Later)."""
    await _get_owned_session(session_id, user, {"_id": 0, "user_id": 1})
    result = await db.fir_sessions.update_one(
        {"session_id": session_id},
        {"$set": {"status": "paused", "updated_at": _dt.now(timezone.utc).isoformat()}},
    )
    if result.matched_count == 0:
        raise HTTPException(404, "Session not found")
    return {"ok": True, "status": "paused"}


# ── v3.3: Back navigation ─────────────────────────────────────────────────────
@router.post("/fir/session/{session_id}/back")
async def fir_back_session(session_id: str, user: Optional[dict] = Depends(current_user_optional)):
    """Go back one probe step, restoring the previous question and clearing its answer."""
    from fir_engine import PROBE_Q
    session = await _get_owned_session(session_id, user)

    probe_history = list(session.get("probe_history", []))
    if not probe_history:
        return {"ok": False, "message": "Already at the beginning"}

    # Pop last answered probe
    last_entry = probe_history.pop()
    probe = last_entry.get("probe")
    slot_key = last_entry.get("slot")

    # Restore slot to previous value
    slots = dict(session.get("slots", {}))
    if slot_key:
        slots[slot_key] = last_entry.get("value")  # restore (may be None)

    # Put current probe back into pending queue, make last_probe the current
    current = session.get("current_probe")
    pending = list(session.get("pending_probes", []))
    if current:
        pending.insert(0, current)

    now = _dt.now(timezone.utc).isoformat()
    await db.fir_sessions.update_one(
        {"session_id": session_id},
        {"$set": {
            "probe_history": probe_history,
            "current_probe": probe,
            "pending_probes": pending,
            "slots": slots,
            "stage": "probe",
            "updated_at": now,
        }}
    )
    pdef = PROBE_Q.get(probe, {}) if probe else {}
    return {
        "ok": True,
        "probe_key": probe,
        "bot_message": pdef.get("message", "Previous question:"),
        "input_type": pdef.get("input_type", "text"),
        "quick_replies": pdef.get("quick_replies", []),
        "skip_label": pdef.get("skip_label"),
        "stage": "probe",
    }


# ── v3.3: Delete evidence file ────────────────────────────────────────────────
@router.delete("/fir/session/{session_id}/evidence/{file_id}")
async def fir_delete_evidence(session_id: str, file_id: str, user: Optional[dict] = Depends(current_user_optional)):
    """Remove an evidence file from a session."""
    from fir_storage import delete_evidence as _del_ev
    session = await _get_owned_session(session_id, user)
    ev_list = session.get("evidence_files", [])
    ev = next((e for e in ev_list if e.get("file_id") == file_id), None)
    if not ev:
        raise HTTPException(404, "File not found")
    # Best-effort delete from object storage
    try:
        await _del_ev(ev.get("storage_path", ""))
    except Exception as exc:
        logger.warning(f"[fir_delete_evidence] storage delete failed: {exc}")
    # Remove from DB
    await db.fir_sessions.update_one(
        {"session_id": session_id},
        {"$pull": {"evidence_files": {"file_id": file_id}},
         "$set": {"updated_at": _dt.now(timezone.utc).isoformat()}},
    )
    return {"ok": True, "deleted_file_id": file_id}


# ── Issue 17: Update evidence caption ─────────────────────────────────────────
class EvidenceCaptionBody(BaseModel):
    caption: str = ""

@router.patch("/fir/session/{session_id}/evidence/{file_id}/caption")
async def fir_update_evidence_caption(session_id: str, file_id: str, body: EvidenceCaptionBody, user: Optional[dict] = Depends(current_user_optional)):
    """Save a caption for an evidence file in the session."""
    session = await _get_owned_session(session_id, user)
    ev_list = session.get("evidence_files", [])
    if not any(e.get("file_id") == file_id for e in ev_list):
        raise HTTPException(404, "Evidence file not found")
    await db.fir_sessions.update_one(
        {"session_id": session_id, "evidence_files.file_id": file_id},
        {"$set": {
            "evidence_files.$.caption": body.caption,
            "updated_at": _dt.now(timezone.utc).isoformat(),
        }},
    )
    return {"ok": True, "file_id": file_id, "caption": body.caption}


# ── v3.3: PDF export ──────────────────────────────────────────────────────────
@router.get("/fir/session/{session_id}/draft.pdf")
async def fir_get_pdf(session_id: str, user: Optional[dict] = Depends(current_user_optional)):
    """Generate and return a PDF of the FIR draft."""
    import io
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.units import cm
    from fastapi.responses import Response as _Resp

    session = await _get_owned_session(session_id, user)
    draft = (session.get("draft") or "").strip()
    if not draft:
        raise HTTPException(400, "No draft generated yet. Please complete the interview first.")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        rightMargin=2 * cm, leftMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DharaTitle", parent=styles["Heading1"],
        fontSize=15, spaceAfter=8, leading=20,
    )
    body_style = ParagraphStyle(
        "DharaBody", parent=styles["Normal"],
        fontSize=10.5, leading=16, spaceAfter=6,
    )
    warn_style = ParagraphStyle(
        "DharaWarn", parent=styles["Normal"],
        fontSize=9, leading=14, textColor=(0.6, 0, 0), spaceAfter=4,
    )

    story = [
        Paragraph("DHARA — FIR Citizen Draft", title_style),
        Paragraph(
            "<i>⚠ This is a citizen draft — NOT a registered FIR. "
            "Present this document at the nearest police station.</i>",
            warn_style,
        ),
        Spacer(1, 0.4 * cm),
    ]
    for line in draft.split("\n"):
        safe = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        if safe.strip():
            story.append(Paragraph(safe, body_style))
        else:
            story.append(Spacer(1, 0.25 * cm))

    # ── Issue 17: Evidence Annex with captions ────────────────────────────────
    evidence_files = list(session.get("evidence_files", []))
    if evidence_files:
        heading_style = ParagraphStyle(
            "DharaHeading", parent=styles["Heading2"],
            fontSize=11, spaceBefore=12, spaceAfter=6, leading=16,
        )
        story.append(Spacer(1, 0.3 * cm))
        story.append(Paragraph("EVIDENCE ANNEX", heading_style))
        for i, ev in enumerate(evidence_files):
            fname = (ev.get("filename") or "File").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            caption = (ev.get("caption") or "").strip()
            ftype = ev.get("file_type", "document")
            if caption:
                exhibit_line = f"<b>Exhibit {i+1}:</b> {fname} — {caption}"
            else:
                exhibit_line = f"<b>Exhibit {i+1}:</b> {fname} ({ftype})"
            story.append(Paragraph(exhibit_line, body_style))

    doc.build(story)
    buf.seek(0)
    return _Resp(
        content=buf.read(),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="FIR_Draft_{session_id[:8]}.pdf"',
            "Cache-Control": "no-cache",
        },
    )


# ── Voter Roll PDF ─────────────────────────────────────────────────────────────
class VoterPdfRequest(BaseModel):
    form_type: str          # 'form6' or 'form8'
    answers: dict           # field_name -> value
    language: str = "en"   # 'en' | 'hi' | 'mr'


@router.post("/voter/pdf")
async def voter_generate_pdf(body: VoterPdfRequest):
    """Generate a Form 6 or Form 8 voter roll application PDF using the existing reportlab engine."""
    import io
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
    from reportlab.lib.units import cm
    from reportlab.lib.colors import HexColor
    from fastapi.responses import Response as _Resp
    import datetime

    if body.form_type not in ("form6", "form8"):
        raise HTTPException(400, "form_type must be 'form6' or 'form8'")

    lang = body.language if body.language in ("en", "hi", "mr") else "en"
    answers = body.answers

    # ── Labels by language ────────────────────────────────────────────────────
    form_titles = {
        "form6": {
            "en": "Form 6 — Application for Inclusion of Name in Electoral Roll",
            "hi": "फॉर्म 6 — मतदाता सूची में नाम जोड़ने का आवेदन",
            "mr": "फॉर्म 6 — मतदार यादीत नाव समाविष्ट करण्याचा अर्ज",
        },
        "form8": {
            "en": "Form 8 — Application for Correction / Shifting of Voter Registration",
            "hi": "फॉर्म 8 — मतदाता पंजीकरण सुधार / स्थानांतरण आवेदन",
            "mr": "फॉर्म 8 — मतदार नोंदणी दुरुस्ती / बदली अर्ज",
        },
    }

    checklist_label = {"en": "Documents to Attach", "hi": "संलग्न दस्तावेज़", "mr": "जोडावयाचे दस्तऐवज"}
    sub_note_label  = {"en": "Submission Note", "hi": "जमा करने की जानकारी", "mr": "सादरीकरण नोट"}
    field_labels_f6 = {
        "en": {
            "applicant_name_en": "Full Name",
            "father_husband_name": "Father's / Husband's Name",
            "date_of_birth": "Date of Birth",
            "gender": "Gender",
            "house_number": "House / Flat Number",
            "street_area": "Street / Area / Village",
            "district_state": "District and State",
            "pin_code": "PIN Code",
            "mobile_number": "Mobile Number",
            "declaration": "Declaration",
        },
        "hi": {
            "applicant_name_en": "पूरा नाम",
            "father_husband_name": "पिता / पति का नाम",
            "date_of_birth": "जन्म तिथि",
            "gender": "लिंग",
            "house_number": "मकान / फ्लैट नंबर",
            "street_area": "गली / क्षेत्र / गाँव",
            "district_state": "जिला और राज्य",
            "pin_code": "पिन कोड",
            "mobile_number": "मोबाइल नंबर",
            "declaration": "घोषणा",
        },
        "mr": {
            "applicant_name_en": "पूर्ण नाव",
            "father_husband_name": "वडिलांचे / पतीचे नाव",
            "date_of_birth": "जन्म तारीख",
            "gender": "लिंग",
            "house_number": "घर / फ्लॅट नंबर",
            "street_area": "रस्ता / परिसर / गाव",
            "district_state": "जिल्हा आणि राज्य",
            "pin_code": "पिन कोड",
            "mobile_number": "मोबाईल नंबर",
            "declaration": "घोषणा",
        },
    }
    field_labels_f8 = {
        "en": {
            "applicant_name_en": "Full Name",
            "change_type": "Change Type",
            "current_epic": "Current EPIC / Voter ID",
            "new_address": "New Address",
            "correction_details": "Correction Details",
            "declaration": "Declaration",
        },
        "hi": {
            "applicant_name_en": "पूरा नाम",
            "change_type": "परिवर्तन का प्रकार",
            "current_epic": "वर्तमान EPIC / वोटर ID",
            "new_address": "नया पता",
            "correction_details": "सुधार विवरण",
            "declaration": "घोषणा",
        },
        "mr": {
            "applicant_name_en": "पूर्ण नाव",
            "change_type": "बदलाचा प्रकार",
            "current_epic": "सध्याचा EPIC / मतदार ID",
            "new_address": "नवीन पत्ता",
            "correction_details": "दुरुस्ती तपशील",
            "declaration": "घोषणा",
        },
    }
    field_labels = field_labels_f6[lang] if body.form_type == "form6" else field_labels_f8[lang]

    checklists = {
        "form6": {
            "en": [
                "Proof of age (Aadhaar / birth certificate / school leaving certificate)",
                "Proof of residence (Aadhaar / utility bill / bank passbook)",
                "Recent passport-size photograph",
                "Completed Form 6 (this document)",
            ],
            "hi": [
                "आयु प्रमाण (आधार / जन्म प्रमाण पत्र / विद्यालय छोड़ने का प्रमाण पत्र)",
                "निवास प्रमाण (आधार / बिजली बिल / बैंक पासबुक)",
                "हालिया पासपोर्ट आकार की फोटो",
                "भरा हुआ फॉर्म 6 (यह दस्तावेज़)",
            ],
            "mr": [
                "वयाचा पुरावा (आधार / जन्म दाखला / शाळा सोडल्याचा दाखला)",
                "निवासाचा पुरावा (आधार / वीज बिल / बँक पासबुक)",
                "अलीकडील पासपोर्ट आकाराचा फोटो",
                "भरलेला फॉर्म 6 (हा दस्तऐवज)",
            ],
        },
        "form8": {
            "en": [
                "Current Voter ID card / EPIC (copy)",
                "Proof of new address (Aadhaar / utility bill)",
                "Supporting document for correction (Aadhaar / PAN / birth certificate)",
                "Recent passport-size photograph",
                "Completed Form 8 (this document)",
            ],
            "hi": [
                "वर्तमान वोटर ID / EPIC (प्रति)",
                "नए पते का प्रमाण (आधार / बिजली बिल)",
                "सुधार के लिए समर्थन दस्तावेज़",
                "हालिया पासपोर्ट आकार की फोटो",
                "भरा हुआ फॉर्म 8 (यह दस्तावेज़)",
            ],
            "mr": [
                "सध्याचे मतदार ID / EPIC (प्रत)",
                "नवीन पत्त्याचा पुरावा (आधार / वीज बिल)",
                "दुरुस्तीसाठी सहाय्यक दस्तऐवज",
                "अलीकडील पासपोर्ट आकाराचा फोटो",
                "भरलेला फॉर्म 8 (हा दस्तऐवज)",
            ],
        },
    }
    submission_notes = {
        "form6": {
            "en": "Submit this form with the above documents to your local Electoral Registration Officer (ERO) or upload at voters.eci.gov.in",
            "hi": "यह फॉर्म दस्तावेजों के साथ स्थानीय ERO को जमा करें या voters.eci.gov.in पर अपलोड करें।",
            "mr": "हा फॉर्म दस्तऐवजांसह स्थानिक ERO कडे जमा करा किंवा voters.eci.gov.in वर अपलोड करा.",
        },
        "form8": {
            "en": "Submit this form with the above documents to your Electoral Registration Officer (ERO) or upload at voters.eci.gov.in",
            "hi": "यह फॉर्म दस्तावेजों के साथ ERO को जमा करें या voters.eci.gov.in पर अपलोड करें।",
            "mr": "हा फॉर्म दस्तऐवजांसह ERO कडे जमा करा किंवा voters.eci.gov.in वर अपलोड करा.",
        },
    }

    # ── Build PDF ─────────────────────────────────────────────────────────────
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        rightMargin=2.2 * cm, leftMargin=2.2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
    )
    styles = getSampleStyleSheet()
    navy = HexColor("#1B2B5B")
    gold = HexColor("#C9973A")

    title_style = ParagraphStyle(
        "VTitle", parent=styles["Heading1"],
        fontSize=14, spaceAfter=6, leading=20, textColor=navy,
    )
    warn_style = ParagraphStyle(
        "VWarn", parent=styles["Normal"],
        fontSize=9, leading=14, textColor=HexColor("#8B0000"), spaceAfter=8,
        borderPad=4,
    )
    field_key_style = ParagraphStyle(
        "VKey", parent=styles["Normal"],
        fontSize=9, leading=13, textColor=HexColor("#666666"), spaceAfter=1,
    )
    field_val_style = ParagraphStyle(
        "VVal", parent=styles["Normal"],
        fontSize=11, leading=16, spaceAfter=8, textColor=HexColor("#111111"),
    )
    section_style = ParagraphStyle(
        "VSection", parent=styles["Heading2"],
        fontSize=10, leading=16, spaceBefore=10, spaceAfter=4, textColor=navy,
    )
    checklist_style = ParagraphStyle(
        "VCheck", parent=styles["Normal"],
        fontSize=10, leading=16, spaceAfter=4, leftIndent=12,
    )
    note_style = ParagraphStyle(
        "VNote", parent=styles["Normal"],
        fontSize=9.5, leading=15, spaceAfter=4, textColor=HexColor("#444444"),
    )
    footer_style = ParagraphStyle(
        "VFooter", parent=styles["Normal"],
        fontSize=8, leading=12, textColor=HexColor("#888888"),
    )

    ts = datetime.datetime.utcnow().strftime("%d %b %Y, %H:%M UTC")
    applicant_name = str(answers.get("applicant_name_en", "Applicant")).replace("&", "&amp;")

    story = [
        # MANDATORY DISCLAIMER — page 1
        Paragraph(
            "⚠ <b>IMPORTANT NOTICE:</b> DHARA has prepared this draft to help you. "
            "This document has not been submitted to the Election Commission of India. "
            "You must submit it yourself at voters.eci.gov.in or at your local ERO office.",
            warn_style,
        ),
        HRFlowable(width="100%", thickness=1, color=gold, spaceAfter=10),
        Paragraph(form_titles[body.form_type][lang].replace("&", "&amp;"), title_style),
        Paragraph(f"Prepared by DHARA · {ts} · For: {applicant_name}", footer_style),
        Spacer(1, 0.4 * cm),
        HRFlowable(width="100%", thickness=0.5, color=HexColor("#DDDDDD"), spaceAfter=10),
    ]

    # ── Field values ──────────────────────────────────────────────────────────
    for field, label in field_labels.items():
        val = answers.get(field)
        if val and str(val).strip():
            safe_label = label.replace("&", "&amp;")
            safe_val = str(val).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            story.append(Paragraph(safe_label.upper(), field_key_style))
            story.append(Paragraph(safe_val, field_val_style))

    story.append(Spacer(1, 0.3 * cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor("#DDDDDD"), spaceAfter=8))

    # ── Checklist ─────────────────────────────────────────────────────────────
    story.append(Paragraph(checklist_label[lang].upper(), section_style))
    for item in checklists[body.form_type][lang]:
        safe_item = item.replace("&", "&amp;")
        story.append(Paragraph(f"☐  {safe_item}", checklist_style))

    story.append(Spacer(1, 0.3 * cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor("#DDDDDD"), spaceAfter=8))

    # ── Submission note ───────────────────────────────────────────────────────
    story.append(Paragraph(sub_note_label[lang].upper(), section_style))
    story.append(Paragraph(
        submission_notes[body.form_type][lang].replace("&", "&amp;"),
        note_style,
    ))

    story.append(Spacer(1, 0.8 * cm))
    story.append(Paragraph(
        "© DHARA · Prepared by Calvil Technologies · This is a citizen assistance draft, "
        "not an official government document.",
        footer_style,
    ))

    doc.build(story)
    buf.seek(0)

    name_slug = "".join(c for c in applicant_name if c.isalnum() or c in " _-")[:30].strip().replace(" ", "_")
    ts_slug = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M")
    filename_prefix = "DHARA_Form6" if body.form_type == "form6" else "DHARA_Form8"
    filename = f"{filename_prefix}_{name_slug}_{ts_slug}.pdf"

    return _Resp(
        content=buf.read(),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-cache",
            "Access-Control-Expose-Headers": "Content-Disposition",
        },
    )


