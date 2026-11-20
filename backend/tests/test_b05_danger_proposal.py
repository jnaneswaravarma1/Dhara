"""JN-01 (B-05 rework) on top of emergent-workspace-9/11.

Proposed tests only. Romanised Hindi/Marathi phrases need native-speaker review
before merge. KNOWN_GAPS are messages that SHOULD be danger but are still
missed: they are xfail so the gap is recorded and runnable, and an xfail that
starts passing (XPASS, strict) tells us the gap has been closed.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.chat_router import _is_immediate_danger  # noqa: E402

NEW_DANGER = [
    "mera pati mujhe maar raha hai",
    "pati maar raha hai",
    "ata amhala dhamki det aahe",
    "mala maarat aahe",
    # first-person keeps danger mode even inside a legal-sounding question
    "what is the punishment if he is hitting me right now",
]
NEW_NOT_DANGER = [
    "what is the punishment for beating someone right now",
    "punishment for kill attempt happening now in the news",
    "is it a crime to hit someone right now",
    "what happens if someone breaks into a house",
]
# Recorded remaining gap (checked against emergent-workspace-9/11 + this change).
KNOWN_GAPS = [
    "someone broke into my house and is inside",       # past tense, intruder still present
    "mera pati mujhe pit raha hai",                    # romanised spelling variant (pit vs peet)
    "ami biponne achi, amake bachan",                  # romanised Bengali
    "আমাকে বাঁচান, আমি বিপদে আছি",                      # Bengali script
    "నన్ను కాపాడండి, నేను ప్రమాదంలో ఉన్నాను",           # Telugu
    "ನನ್ನನ್ನು ರಕ್ಷಿಸಿ, ನಾನು ಅಪಾಯದಲ್ಲಿದ್ದೇನೆ",         # Kannada
    "મને બચાવો, હું જોખમમાં છું",                       # Gujarati
]


@pytest.mark.parametrize("msg", NEW_DANGER)
def test_b05_new_danger_recognised(msg):
    assert _is_immediate_danger(msg), msg


@pytest.mark.parametrize("msg", NEW_NOT_DANGER)
def test_b05_new_legal_questions_not_danger(msg):
    assert not _is_immediate_danger(msg), msg


@pytest.mark.xfail(strict=True, reason="B-05 remaining gap: still missed")
@pytest.mark.parametrize("msg", KNOWN_GAPS)
def test_b05_known_gaps_still_missed(msg):
    assert _is_immediate_danger(msg), msg
