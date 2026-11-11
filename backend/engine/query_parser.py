"""Layer A — Query Parser.

Parses every user query into a LegalQuery object before any retrieval.
Uses a fast LLM (Claude Haiku) with structured JSON output.
Falls back to a minimal LegalQuery on timeout or parse error.
"""
from __future__ import annotations
import asyncio
import json
import re
import uuid
import logging
from typing import Optional

from services.ai_gateway import send_once, QUERY_PARSE_SPEC

from engine.models import LegalQuery, Jurisdiction

logger = logging.getLogger("gandhikar")

# ── Deterministic imperative-prefix stripper ─────────────────────────────────
# Many users phrase legal questions as commands ("help me file an FIR",
# "please help me understand domestic violence law", "can you explain
# property rights").  The imperative prefix carries no legal content and
# confuses both the LLM query parser AND the corpus embedding similarity.
# We strip these before feeding the query to the LLM and before the
# fallback corpus retrieval, while preserving original_query intact for
# logging and LLM context.

_IMPERATIVE_PREFIX_RE = re.compile(
    r"^(?:"
    # "help [me/us] [,] [to/with/understand]" — me/us or to/with REQUIRED after "help"
    # so "help desk number" (no me/us/to/with) is never stripped.
    # [,\s]+ handles "help me, my husband…" (comma after "me").
    r"(?:please\s+)?help\s+(?:(?:me|us)[,\s]+(?:to\s+|with\s+(?:my\s+)?|understand\s+|know\s+(?:about\s+)?)?|to\s+|with\s+(?:my\s+)?)"
    r"|"
    # "can/could you [please] [help [me] [to/with] / tell me [about] / explain]"
    r"(?:can|could)\s+you\s+(?:please\s+)?(?:help\s+(?:me\s+)?(?:to\s+|with\s+)?|tell\s+me\s+(?:about\s+)?|explain\s+(?:to\s+me\s+)?)?"
    r"|"
    # "I want to [know [about] / understand]"
    r"i\s+want\s+to\s+(?:know\s+(?:about\s+)?|understand\s+)?"
    r"|"
    # "I need to [know [about] / understand]"
    r"i\s+need\s+to\s+(?:know\s+(?:about\s+)?|understand\s+)?"
    r"|"
    # "I need help [with [understanding]]"
    r"i\s+need\s+help\s+(?:with\s+(?:understanding\s+)?)?"
    r"|"
    # "please [tell me [about] / explain [to me]]"
    r"please\s+(?:tell\s+me\s+(?:about\s+)?|explain\s+(?:to\s+me\s+)?)?"
    r")",
    re.IGNORECASE,
)


def strip_imperative_prefix(query: str) -> str:
    """Remove a leading imperative/help prefix from a legal query.

    Returns the stripped text when at least 2 words remain; otherwise
    returns the original unchanged (safety guard against stripping too much).

    Examples::

        strip_imperative_prefix("help me file an FIR")
        # → "file an FIR"

        strip_imperative_prefix("please help me understand domestic violence law")
        # → "domestic violence law"

        strip_imperative_prefix("can you explain property rights?")
        # → "property rights?"

        strip_imperative_prefix("i need to know about cheque bounce")
        # → "cheque bounce"

        strip_imperative_prefix("help me")       # bare — keep unchanged
        # → "help me"
    """
    stripped = _IMPERATIVE_PREFIX_RE.sub("", query).strip()
    # Keep at least 2 words — avoid returning an empty or single-word stump
    if len(stripped.split()) >= 2:
        return stripped
    return query

_CANONICAL_ACTIONS = (
    "enter,exit,refuse_entry,restrict_access,evict,arrest,detain,bail,search,seize,release,"
    "hit,assault,threaten,intimidate,hurt,kill,trespass,damage,demolish,destroy,defile,"
    "desecrate,disrupt_worship,change_religious_character,convert_religion,"
    "publish,broadcast,defame,insult,incite,hate_speech,"
    "record,photograph,surveil,share_data,collect_data,"
    "sell,buy,rent,lease,transfer,mortgage,inherit,"
    "marry,divorce,separate,adopt,custody,maintenance,"
    "employ,dismiss,terminate,suspend,discriminate,"
    "vote,register_voter,contest_election,campaign,"
    "file_complaint,file_fir,file_petition,appeal,"
    "pay,withhold_payment,cheat,fraud,breach_contract,"
    "drive,park,speed,traffic_violation,"
    "construct,build,alter,develop,encroach,"
    "pollute,dump_waste,deforest"
)

_QUERY_PARSER_SYSTEM = f"""You are a legal query parser for an Indian law system.
Parse the user's query into structured JSON. Prioritize ACTIONS above all else.

CANONICAL ACTIONS (use ONLY these; pick the closest match):
{_CANONICAL_ACTIONS}

LOCATION TYPES: place_of_worship, private_property, public_road, workplace,
court, police_station, government_office, hospital, home, market, religious_place

CRITICAL RULE: action "enter" + location_type "place_of_worship" does NOT mean
domain "change_religious_character". Location is context. ACTION determines domain.

Output ONLY valid JSON matching this schema (no prose, no markdown):
{{
  "original_query": "<user's exact words>",
  "normalized_query": "<English restatement in 1 sentence>",
  "language": "<detected: en|hi|mr|ta|te|bn|gu|kn|ml|pa|or|auto>",
  "jurisdiction": {{"country": "India", "state": null, "district": null}},
  "event_date": null,
  "actions": ["<action from canonical list>"],
  "actors": ["<person or entity performing the action>"],
  "actor_roles": ["<e.g. landlord, police, employer, husband — whoever PERFORMS the action above>"],
  "asking_as_role": "<ONLY if the user explicitly frames themselves as a specific party in the situation — e.g. tenant, employee, bystander, journalist, victim, patient, customer, citizen, student. This is the USER's OWN role, which is often DIFFERENT from actor_roles above (e.g. a tenant asking about a LANDLORD's duties). null if not stated or unclear.>",
  "objects": ["<thing acted upon>"],
  "locations": ["<place name>"],
  "location_types": ["<type from list above>"],
  "intent": ["<purpose behind the action>"],
  "harm": ["<type of harm: physical|financial|emotional|legal|religious>"],
  "candidate_issues": ["<legal issue: e.g. trespass, arrest_rights, dowry>"],
  "missing_material_facts": ["<facts that would change the legal answer>"],
  "urgency": "immediate|standard|informational",
  "requested_outcome": "<what the user wants to know or achieve>"
}}"""


async def parse_query(
    query: str,
    language: str = "en",
    llm_key: str = "",
    prior_state: Optional[LegalQuery] = None,
) -> LegalQuery:
    """Run Layer A. Returns a LegalQuery. Never raises — falls back to minimal object."""

    # Strip common imperative prefixes so the LLM receives a cleaner seed
    # and the corpus fallback (below) also benefits if the LLM times out.
    # original_query is always the raw user text — strip only affects normalized.
    clean_query = strip_imperative_prefix(query)

    # Build the prompt, including prior context if this is a continuation turn
    prompt = clean_query
    if prior_state and prior_state.candidate_issues:
        prompt = (
            f"[PRIOR CONTEXT: issues={prior_state.candidate_issues}, "
            f"actions={prior_state.actions}, actors={prior_state.actor_roles}]\n"
            f"NEW USER MESSAGE: {clean_query}"
        )

    async def _attempt() -> Optional[LegalQuery]:
        try:
            # Routed through the AI gateway (model/key/timeout come from
            # services/ai_gateway.QUERY_PARSE_SPEC; `llm_key` is kept in the
            # signature for backward compatibility but the gateway resolves the key).
            raw = await send_once(
                f"qparse-{uuid.uuid4().hex[:8]}",
                _QUERY_PARSER_SYSTEM,
                prompt,
                spec=QUERY_PARSE_SPEC,
                caller="query_parser",
                timeout_s=QUERY_PARSE_SPEC.timeout_s,
            )
            raw_str = str(raw or "").strip()

            # Extract JSON block
            start = raw_str.find("{")
            end = raw_str.rfind("}") + 1
            if start < 0 or end <= start:
                logger.warning("[qparser] no JSON in response: %r", raw_str[:80])
                return None

            data = json.loads(raw_str[start:end])
            # Normalise jurisdiction
            j = data.get("jurisdiction") or {}
            data["jurisdiction"] = {
                "country": j.get("country", "India"),
                "state": j.get("state"),
                "district": j.get("district"),
            }
            lq = LegalQuery.model_validate(data)
            if not lq.language or lq.language == "auto":
                lq.language = language
            logger.info("[qparser] ok | actions=%s | domain_candidates=%s",
                        lq.actions[:3], lq.candidate_issues[:3])
            return lq

        except asyncio.TimeoutError:
            logger.warning("[qparser] timeout(10s) | q=%r", query[:60])
        except json.JSONDecodeError as e:
            logger.warning("[qparser] JSON parse error: %s", e)
        except Exception as e:
            logger.warning("[qparser] error: %s", e)
        return None

    result = await _attempt()
    if result:
        # If the LLM echo'd back the raw query unchanged (no meaningful rewrite),
        # apply the deterministic prefix-stripper so the corpus gets a clean seed.
        if result.normalized_query in (query, clean_query):
            result.normalized_query = clean_query
        # Always preserve the real user text as original_query for audit/logging.
        result.original_query = query
        return result

    # Minimal fallback — do not raise
    logger.warning("[qparser] using minimal fallback for q=%r", query[:60])
    return LegalQuery(
        original_query=query,
        normalized_query=clean_query,   # stripped version for corpus retrieval
        language=language,
        jurisdiction=Jurisdiction(),
        actions=[],
        candidate_issues=[],
        urgency="standard",
        requested_outcome="unknown",
    )
