"""B-05 proposal: wider danger recognition and fewer legal-question false alarms.

Proposed tests only. The phrases for Hindi/Marathi (romanised) need review by
native speakers before this is merged. Add 20 danger + 20 look-alike legal
messages per Wave 1 language as the workbook asks.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.chat_router import _is_immediate_danger  # noqa: E402

NEW_DANGER = [
    "someone is breaking into my house",
    "somebody is trying to break in",
    "they just broke into my house",
    "mera pati mujhe maar raha hai",
    "pati maar raha hai",
    "ata amhala dhamki det aahe",
    "mala maarat aahe",
    # first-person keeps the old behaviour even inside a legal-sounding question
    "what is the punishment if he is hitting me right now",
]
NEW_NOT_DANGER = [
    "what is the punishment for beating someone right now",
    "punishment for kill attempt happening now in the news",
    "is it a crime to hit someone right now",
    "what happens if someone breaks into a house",
    "someone broke into my house last year, what can I claim",
]


@pytest.mark.parametrize("msg", NEW_DANGER)
def test_b05_new_danger_recognised(msg):
    assert _is_immediate_danger(msg), msg


@pytest.mark.parametrize("msg", NEW_NOT_DANGER)
def test_b05_new_legal_questions_not_danger(msg):
    assert not _is_immediate_danger(msg), msg
