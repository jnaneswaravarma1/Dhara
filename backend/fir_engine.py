"""
DHARA FIR Interview Engine — v3.2 (GPS + Evidence + Quality Fixes)
Conversational, stage-based FIR drafting assistant.
Fresh session isolation: every session is 100% blank — no carryover.

v3.1 fixes (2026-09):
  1. Date contradiction detection: "today" vs explicit extracted date.
  2. Place validation: off-topic answers go to parked_questions, re-ask place.
  3. parked_questions: new field, populated on off-topic probe answers.
  4. Accused slot guard: witness-about-accused answers rerouted to witnesses.
  5. Witness follow-up: vague witness answers trigger name/detail probe.
  6. Phone validation: placeholder / non-numeric phone → one retry.
  7. Section mapping fix: BNS 77/78 removed from general harassment.
  8. Incident type fallback: keyword classifier when LLM returns only ["other"].

v3.2 fixes (2026-09):
  5b. Vague witness detection: now catches answers with no names/numbers regardless of length.
  9.  Citation Guard: dropped_sections written to DB with reason_dropped field.
      suggest_sections() now returns (confirmed, dropped) tuple.
"""
from __future__ import annotations

import uuid
import json
import re
import asyncio
from datetime import datetime, timezone, date, timedelta
from typing import Optional
import httpx
import logging

from services.ai_gateway import send_once, FIR_EXTRACT_SPEC, FIR_CORRECTION_SPEC, FIR_DRAFT_SPEC

logger = logging.getLogger("fir_engine")

# ─── v3.3: Multilingual translation system ────────────────────────────────────
# All static bot messages, translated into hi / mr / ta (falls back to 'en').

_STATIC_TRANS: dict[str, dict[str, str]] = {
    "safe_prompt": {
        "en": "Before we begin: are you safe right now?",
        "hi": "शुरू करने से पहले: क्या आप अभी सुरक्षित हैं?",
        "mr": "सुरू करण्यापूर्वी: तुम्ही आत्ता सुरक्षित आहात का?",
        "ta": "தொடங்குவதற்கு முன்: நீங்கள் இப்போது பாதுகாப்பாக இருக்கிறீர்களா?",
        "te": "ప్రారంభించే ముందు: మీరు ఇప్పుడు సురక్షితంగా ఉన్నారా?",
        "kn": "ಪ್ರಾರಂಭಿಸುವ ಮೊದಲು: ನೀವು ಈಗ ಸುರಕ್ಷಿತವಾಗಿದ್ದೀರಾ?",
    },
    "safe_yes": {
        "en": "Yes, I'm safe",
        "hi": "हाँ, मैं सुरक्षित हूँ",
        "mr": "होय, मी सुरक्षित आहे",
        "ta": "ஆம், நான் பாதுகாப்பாக இருக்கிறேன்",
        "te": "అవును, నేను సురక్షితంగా ఉన్నాను",
        "kn": "ಹೌದು, ನಾನು ಸುರಕ್ಷಿತ",
    },
    "safe_no": {
        "en": "No, I need help",
        "hi": "नहीं, मुझे मदद चाहिए",
        "mr": "नाही, मला मदत हवी आहे",
        "ta": "இல்லை, எனக்கு உதவி வேண்டும்",
        "te": "లేదు, నాకు సహాయం కావాలి",
        "kn": "ಇಲ್ಲ, ನನಗೆ ಸಹಾಯ ಬೇಕು",
    },
    "not_safe_msg": {
        "en": (
            "\u26a0\ufe0f Please stay safe first.\n\n"
            "\U0001f4de Emergency: 112\n\U0001f4de Women Helpline: 181\n"
            "\U0001f4de Ambulance: 108\n\U0001f4de NALSA Legal Aid: 15100 (free)\n\n"
            "Call for help now if you are in immediate danger.\n\n"
            "When you are safe, tap the button below to continue filing your complaint."
        ),
        "hi": (
            "\u26a0\ufe0f पहले सुरक्षित रहें।\n\n"
            "\U0001f4de आपातकाल: 112\n\U0001f4de महिला हेल्पलाइन: 181\n"
            "\U0001f4de एम्बुलेंस: 108\n\U0001f4de NALSA निःशुल्क सहायता: 15100\n\n"
            "यदि आप तत्काल खतरे में हैं तो अभी मदद के लिए कॉल करें।\n\n"
            "जब सुरक्षित हों, शिकायत जारी रखने के लिए नीचे बटन दबाएं।"
        ),
        "mr": (
            "\u26a0\ufe0f प्रथम सुरक्षित राहा.\n\n"
            "\U0001f4de आणीबाणी: 112\n\U0001f4de महिला हेल्पलाइन: 181\n"
            "\U0001f4de रुग्णवाहिका: 108\n\U0001f4de NALSA मोफत मदत: 15100\n\n"
            "तात्काळ धोक्यात असल्यास आत्ता मदतीसाठी कॉल करा.\n\n"
            "सुरक्षित असाल तेव्हा, तक्रार सुरू ठेवण्यासाठी खाली बटण दाबा."
        ),
        "ta": (
            "\u26a0\ufe0f முதலில் பாதுகாப்பாக இருங்கள்.\n\n"
            "\U0001f4de அவசரநிலை: 112\n\U0001f4de மகளிர் உதவி: 181\n"
            "\U0001f4de ஆம்புலன்ஸ்: 108\n\U0001f4de NALSA இலவச உதவி: 15100\n\n"
            "ஆபத்தில் இருந்தால் உடனே அழையுங்கள்.\n\n"
            "பாதுகாப்பாக இருக்கும்போது, புகாரை தொடர கீழே உள்ள பொத்தானை அழுத்தவும்."
        ),
    },
    "safe_continue": {
        "en": "I'm safe now \u2014 continue filing",
        "hi": "मैं अब सुरक्षित हूँ — शिकायत जारी रखें",
        "mr": "मी आता सुरक्षित आहे — तक्रार सुरू ठेवा",
        "ta": "நான் இப்போது பாதுகாப்பாக இருக்கிறேன் — தொடரவும்",
    },
    "exit_now": {
        "en": "Exit for now",
        "hi": "अभी बाहर जाएं",
        "mr": "आत्ता बाहेर पडा",
        "ta": "இப்போது வெளியேறு",
    },
    "trust_msg": {
        "en": "Thank you for trusting DHARA with your complaint.",
        "hi": "DHARA पर विश्वास करने के लिए धन्यवाद।",
        "mr": "DHARA वर विश्वास ठेवल्याबद्दल धन्यवाद.",
        "ta": "DHARA யை நம்பி புகார் தெரிவித்தமைக்கு நன்றி.",
    },
    "pocso_alert": {
        "en": "\n\U0001f198 This may involve a child — Childline: 1098 (24\u00d77 FREE)",
        "hi": "\n\U0001f198 इसमें एक बच्चे का मामला हो सकता है — Childline: 1098 (24×7 निःशुल्क)",
        "mr": "\n\U0001f198 यात मुलाचा संबंध असू शकतो — Childline: 1098 (24×7 मोफत)",
        "ta": "\n\U0001f198 இதில் குழந்தை சம்பந்தப்பட்டிருக்கலாம் — Childline: 1098 (24×7 இலவசம்)",
    },
    "dv_alert": {
        "en": "\n\U0001f198 Women / DV Helpline: 181 (24\u00d77 FREE)",
        "hi": "\n\U0001f198 महिला / घरेलू हिंसा हेल्पलाइन: 181 (24×7 निःशुल्क)",
        "mr": "\n\U0001f198 महिला / घरगुती हिंसाचार हेल्पलाइन: 181 (24×7 मोफत)",
        "ta": "\n\U0001f198 மகளிர் / குடும்ப வன்முறை உதவி: 181 (24×7 இலவசம்)",
    },
    "what_happened": {
        "en": "\nPlease tell me in your own words: what happened?\nTake your time — speak or type freely.",
        "hi": "\nकृपया अपने शब्दों में बताएं: क्या हुआ?\nजल्दी मत करें — बोलकर या टाइप करके बताएं।",
        "mr": "\nकृपया तुमच्या शब्दांत सांगा: काय झाले?\nघाई करू नका — बोलून किंवा टाइप करून सांगा.",
        "ta": "\nதயவுசெய்து உங்கள் வார்த்தைகளில் சொல்லுங்கள்: என்ன நடந்தது?\nதாராளமாக பேசுங்கள் அல்லது தட்டச்சு செய்யுங்கள்.",
    },
    "safe_now_continue": {
        "en": "I'm glad you're safe. Please tell me in your own words: what happened?\nTake your time — speak or type freely.",
        "hi": "खुशी है कि आप सुरक्षित हैं। कृपया बताएं: क्या हुआ?\nजल्दी मत करें — बोलकर या टाइप करके बताएं।",
        "mr": "तुम्ही सुरक्षित आहात हे ऐकून बरे वाटले. कृपया सांगा: काय झाले?\nघाई करू नका — बोलून किंवा टाइप करून सांगा.",
        "ta": "நீங்கள் பாதுகாப்பாக இருக்கிறீர்கள் என்று மகிழ்ச்சி. என்ன நடந்தது?\nதாராளமாக பேசுங்கள் அல்லது தட்டச்சு செய்யுங்கள்.",
    },
    "more_detail": {
        "en": "Could you share a bit more detail about what happened?",
        "hi": "क्या आप जो हुआ उसके बारे में थोड़ा और विस्तार से बता सकते हैं?",
        "mr": "जे झाले त्याबद्दल थोडे अधिक सांगू शकता का?",
        "ta": "என்ன நடந்தது என்பதை இன்னும் கொஞ்சம் விவரமாக சொல்ல முடியுமா?",
    },
    "emergency_danger": {
        "en": (
            "\u26a0\ufe0f You seem to be describing an ongoing emergency.\n\n"
            "Please call for help FIRST:\n"
            "\U0001f4de Emergency: 112\n\U0001f4de Women Helpline: 181\n\U0001f4de Ambulance: 108\n\n"
            "Your session is saved. Come back to complete your complaint when you are safe."
        ),
        "hi": (
            "\u26a0\ufe0f ऐसा लगता है कि आप एक चल रही आपातस्थिति बता रहे हैं।\n\n"
            "पहले मदद के लिए कॉल करें:\n"
            "\U0001f4de आपातकाल: 112\n\U0001f4de महिला हेल्पलाइन: 181\n\U0001f4de एम्बुलेंस: 108\n\n"
            "आपका सत्र सुरक्षित है। सुरक्षित होने पर वापस आएं।"
        ),
        "mr": (
            "\u26a0\ufe0f असे वाटते की तुम्ही सध्या आणीबाणीची परिस्थिती सांगत आहात.\n\n"
            "आधी मदतीसाठी कॉल करा:\n"
            "\U0001f4de आणीबाणी: 112\n\U0001f4de महिला हेल्पलाइन: 181\n\U0001f4de रुग्णवाहिका: 108\n\n"
            "तुमचे सत्र जतन आहे. सुरक्षित असाल तेव्हा परत या."
        ),
        "ta": (
            "\u26a0\ufe0f நீங்கள் தொடர்ந்து நடக்கும் அவசரநிலையை விவரிப்பது போல் தெரிகிறது.\n\n"
            "முதலில் உதவிக்கு அழையுங்கள்:\n"
            "\U0001f4de அவசரநிலை: 112\n\U0001f4de மகளிர் உதவி: 181\n\U0001f4de ஆம்புலன்ஸ்: 108\n\n"
            "உங்கள் session சேமிக்கப்பட்டுள்ளது. பாதுகாப்பாக இருக்கும்போது திரும்பி வாருங்கள்."
        ),
    },
    "safe_continue_short": {
        "en": "I'm safe \u2014 continue filing",
        "hi": "मैं सुरक्षित हूँ — शिकायत जारी रखें",
        "mr": "मी सुरक्षित आहे — तक्रार सुरू ठेवा",
        "ta": "நான் பாதுகாப்பாக இருக்கிறேன் — தொடரவும்",
    },
    "thanks_sharing": {
        "en": "Thank you for sharing that. I can see this involves {types_str}.\n\nI have a few clarifying questions to complete your complaint.\n\n{suffix}",
        "hi": "साझा करने के लिए धन्यवाद। मैं देख सकता हूँ कि यह {types_str} से संबंधित है।\n\nआपकी शिकायत पूरी करने के लिए कुछ सवाल हैं।\n\n{suffix}",
        "mr": "सांगितल्याबद्दल धन्यवाद. हे {types_str} शी संबंधित आहे.\n\nतक्रार पूर्ण करण्यासाठी काही प्रश्न आहेत.\n\n{suffix}",
        "ta": "பகிர்ந்துகொண்டதற்கு நன்றி. இது {types_str} தொடர்பானது.\n\nபுகாரை முழுமையாக்க சில கேள்விகள் கேட்கிறேன்.\n\n{suffix}",
    },
    "welcome_back": {
        "en": "Welcome back! Continuing from where you left off.\n\n{suffix}",
        "hi": "वापस स्वागत है! जहाँ छोड़ा था वहाँ से जारी रखते हैं।\n\n{suffix}",
        "mr": "परत स्वागत आहे! जिथे सोडले होते तिथून सुरू ठेवूया.\n\n{suffix}",
        "ta": "மீண்டும் வரவேற்கிறோம்! நீங்கள் நிறுத்திய இடத்தில் இருந்து தொடரலாம்.\n\n{suffix}",
    },
    "gps_found": {
        "en": "I found this address:\n\n\U0001f4cd {addr}\n\nIs this correct?",
        "hi": "मुझे यह पता मिला:\n\n\U0001f4cd {addr}\n\nक्या यह सही है?",
        "mr": "मला हा पत्ता सापडला:\n\n\U0001f4cd {addr}\n\nहे बरोबर आहे का?",
        "ta": "இந்த முகவரி கிடைத்தது:\n\n\U0001f4cd {addr}\n\nசரிதானா?",
    },
    "gps_yes": {
        "en": "Yes, that's correct",
        "hi": "हाँ, यह सही है",
        "mr": "होय, हे बरोबर आहे",
        "ta": "ஆம், சரிதான்",
    },
    "gps_no": {
        "en": "No, use text description",
        "hi": "नहीं, टेक्स्ट विवरण का उपयोग करें",
        "mr": "नाही, मजकूर वर्णन वापरा",
        "ta": "இல்லை, உரை விவரத்தை பயன்படுத்துக",
    },
    "place_off_topic": {
        "en": "I've noted your question and will come back to it.\n\nFor the complaint, I need the **location** of the incident.\n{suffix}",
        "hi": "मैंने आपका सवाल नोट कर लिया है और बाद में उस पर वापस आऊंगा।\n\nशिकायत के लिए, मुझे घटना की **जगह** चाहिए।\n{suffix}",
        "mr": "मी तुमचा प्रश्न नोंदला आहे आणि नंतर त्याकडे परत येईन.\n\nतक्रारीसाठी, मला घटनेचे **ठिकाण** हवे आहे.\n{suffix}",
        "ta": "உங்கள் கேள்வியை குறித்துக்கொண்டேன், பின்னர் திரும்பி வருவேன்.\n\nபுகாருக்காக, சம்பவம் நடந்த **இடம்** தேவை.\n{suffix}",
    },
    "section_found": {
        "en": (
            "Based on your description, the following BNS sections appear to apply:\n\n{sec_lines}\n\n"
            "\u26a0\ufe0f These are suggestions only \u2014 the investigating officer determines final sections.\n\n"
            "Shall I proceed with generating your draft?"
        ),
        "hi": (
            "आपके विवरण के आधार पर, निम्नलिखित BNS धाराएं लागू होती हैं:\n\n{sec_lines}\n\n"
            "\u26a0\ufe0f ये केवल सुझाव हैं — जांच अधिकारी अंतिम धाराएं तय करेंगे।\n\n"
            "क्या मैं आपका मसौदा तैयार करूँ?"
        ),
        "mr": (
            "तुमच्या वर्णनावर आधारित, खालील BNS कलम लागू होतात:\n\n{sec_lines}\n\n"
            "\u26a0\ufe0f हे केवळ सुचवणे आहेत — तपास अधिकारी अंतिम कलम ठरवतात.\n\n"
            "मसुदा तयार करू का?"
        ),
        "ta": (
            "உங்கள் விவரணையின் அடிப்படையில், பின்வரும் BNS பிரிவுகள் பொருந்தும்:\n\n{sec_lines}\n\n"
            "\u26a0\ufe0f இவை பரிந்துரைகள் மட்டுமே — விசாரணை அதிகாரி இறுதி பிரிவுகளை தீர்மானிப்பார்.\n\n"
            "மசோதாவை உருவாக்கட்டுமா?"
        ),
    },
    "section_none": {
        "en": (
            "I've gathered all the details for your complaint.\n\n"
            "The applicable BNS sections will be noted by the investigating officer.\n\n"
            "Ready to generate your FIR draft?"
        ),
        "hi": (
            "मैंने आपकी शिकायत के लिए सभी विवरण एकत्र कर लिए हैं।\n\n"
            "लागू BNS धाराएं जांच अधिकारी द्वारा नोट की जाएंगी।\n\n"
            "क्या आप FIR मसौदा तैयार करने के लिए तैयार हैं?"
        ),
        "mr": (
            "मी तुमच्या तक्रारीसाठी सर्व तपशील गोळा केले आहेत.\n\n"
            "लागू BNS कलम तपास अधिकाऱ्याकडून नोंदवले जातील.\n\n"
            "FIR मसुदा तयार करायचा आहे का?"
        ),
        "ta": (
            "உங்கள் புகாருக்கான அனைத்து விவரங்களையும் சேகரித்தேன்.\n\n"
            "பொருந்தும் BNS பிரிவுகளை விசாரணை அதிகாரி குறிப்பிடுவார்.\n\n"
            "FIR மசோதாவை உருவாக்க தயாரா?"
        ),
    },
    "yes_proceed": {
        "en": "Yes, proceed",
        "hi": "हाँ, आगे बढ़ें",
        "mr": "होय, पुढे जा",
        "ta": "ஆம், தொடரவும்",
    },
    "go_back": {
        "en": "Go back",
        "hi": "वापस जाएं",
        "mr": "मागे जा",
        "ta": "திரும்பு",
    },
    "summary_correct": {
        "en": "Here is a summary of your complaint:\n\n{summary}\n\nIs everything correct? Shall I generate your FIR draft?",
        "hi": "यहाँ आपकी शिकायत का सारांश है:\n\n{summary}\n\nक्या सब कुछ सही है? क्या मैं FIR मसौदा तैयार करूँ?",
        "mr": "तुमच्या तक्रारीचा सारांश:\n\n{summary}\n\nसर्व काही बरोबर आहे का? FIR मसुदा तयार करू का?",
        "ta": "உங்கள் புகாரின் சுருக்கம்:\n\n{summary}\n\nஎல்லாம் சரியா? FIR மசோதாவை உருவாக்கட்டுமா?",
    },
    "yes_generate": {
        "en": "Yes, generate my draft",
        "hi": "हाँ, मसौदा तैयार करें",
        "mr": "होय, मसुदा तयार करा",
        "ta": "ஆம், மசோதாவை உருவாக்குக",
    },
    "edit_something": {
        "en": "Edit something",
        "hi": "कुछ बदलें",
        "mr": "काहीतरी बदला",
        "ta": "ஏதாவது திருத்துக",
    },
    "what_to_change": {
        "en": "What would you like to change? Please type the correction and I'll update your complaint.",
        "hi": "आप क्या बदलना चाहते हैं? कृपया सुधार टाइप करें और मैं आपकी शिकायत अपडेट करूंगा।",
        "mr": "तुम्हाला काय बदलायचे आहे? कृपया दुरुस्ती टाइप करा आणि मी तुमची तक्रार अपडेट करेन.",
        "ta": "என்ன மாற்ற விரும்புகிறீர்கள்? திருத்தத்தை தட்டச்சு செய்யுங்கள்.",
    },
    "draft_ready": {
        "en": "\u2705 Your FIR draft is ready!\n\nTap \"View Draft\" to see, save, or export it.",
        "hi": "\u2705 आपका FIR मसौदा तैयार है!\n\n\"View Draft\" दबाकर देखें, सहेजें या निर्यात करें।",
        "mr": "\u2705 तुमचा FIR मसुदा तयार आहे!\n\n\"View Draft\" दाबून पहा, जतन करा किंवा निर्यात करा.",
        "ta": "\u2705 உங்கள் FIR மசோதா தயார்!\n\n\"View Draft\" அழுத்தி காணுங்கள், சேமியுங்கள் அல்லது பகிருங்கள்.",
    },
    "draft_already": {
        "en": "Your draft is ready. Tap \"View Draft\" to see it.",
        "hi": "आपका मसौदा तैयार है। \"View Draft\" दबाकर देखें।",
        "mr": "तुमचा मसुदा तयार आहे. \"View Draft\" दाबून पहा.",
        "ta": "உங்கள் மசோதா தயார். \"View Draft\" அழுத்தி காணுங்கள்.",
    },
    "cybercrime_alert": {
        "en": (
            "\U0001f6a8 IMPORTANT \u2014 CYBERCRIME GOLDEN HOUR ALERT:\n"
            "Call 1930 (National Cybercrime Helpline) IMMEDIATELY if you lost money.\n"
            "The sooner you report, the better your chances of recovering funds.\n"
            "You can also file at cybercrime.gov.in\n\n"
        ),
        "hi": (
            "\U0001f6a8 महत्वपूर्ण \u2014 साइबर अपराध स्वर्णिम घंटा:\n"
            "पैसे खोए हैं तो तुरंत 1930 (राष्ट्रीय साइबर अपराध हेल्पलाइन) पर कॉल करें।\n"
            "जितनी जल्दी रिपोर्ट करें, पैसे वापस मिलने की संभावना उतनी अधिक।\n"
            "cybercrime.gov.in पर भी दर्ज करें।\n\n"
        ),
        "mr": (
            "\U0001f6a8 महत्वाचे \u2014 सायबर गुन्हा गोल्डन अवर:\n"
            "पैसे गेले असतील तर 1930 (राष्ट्रीय सायबर गुन्हा हेल्पलाइन) वर ताबडतोब कॉल करा.\n"
            "cybercrime.gov.in वरही नोंदवा.\n\n"
        ),
        "ta": (
            "\U0001f6a8 முக்கியம் \u2014 சைபர் கிரைம் தங்கநேரம்:\n"
            "பணம் இழந்திருந்தால் 1930 (தேசிய சைபர் கிரைம் உதவி) உடனே அழையுங்கள்.\n"
            "cybercrime.gov.in இலும் பதிவு செய்யலாம்.\n\n"
        ),
    },
}

# Probe question translations (hi / mr / ta only; falls back to English PROBE_Q["message"])
_PROBE_MSG_TRANS: dict[str, dict[str, str]] = {
    "probe_date": {
        "hi": "यह कब हुआ? (तिथि और वर्ष यदि संभव हो)",
        "mr": "हे केव्हा झाले? (तारीख आणि वर्ष शक्य असल्यास)",
        "ta": "இது எப்போது நடந்தது? (தேதி மற்றும் ஆண்டு சாத்தியமாயின்)",
    },
    "probe_time": {
        "hi": "यह लगभग किस समय हुआ?",
        "mr": "हे अंदाजे किती वाजता झाले?",
        "ta": "இது தோராயமாக எத்தனை மணிக்கு நடந்தது?",
    },
    "probe_place_text": {
        "hi": "यह कहाँ हुआ? (क्षेत्र / सड़क / शहर / राज्य)",
        "mr": "हे कुठे झाले? (परिसर / रस्ता / शहर / राज्य)",
        "ta": "இது எங்கே நடந்தது? (பகுதி / தெரு / நகரம் / மாநிலம்)",
    },
    "probe_place_gps": {
        "hi": "क्या आप GPS से सटीक घटना स्थान बताना चाहेंगे?\n(पुलिस को सटीक स्थान पहचानने में मदद करता है — वैकल्पिक)",
        "mr": "तुम्हाला GPS वापरून अचूक घटनेचे ठिकाण दाखवायचे आहे का?\n(पोलिसांना अचूक जागा ओळखण्यास मदत — पर्यायी)",
        "ta": "GPS மூலம் துல்லியமான சம்பவ இடத்தை குறிக்க விரும்புகிறீர்களா?\n(விருப்பத்தேர்வு)",
    },
    "probe_theft_items": {
        "hi": "वास्तव में क्या चोरी हुआ? प्रत्येक वस्तु और अनुमानित मूल्य बताएं।",
        "mr": "नक्की काय चोरी झाले? प्रत्येक वस्तू आणि अंदाजे मूल्य सांगा.",
        "ta": "சரியாக என்ன திருடப்பட்டது? ஒவ்வொரு பொருளையும் அதன் மதிப்பையும் பட்டியலிடுங்கள்.",
    },
    "probe_injury": {
        "hi": "क्या कोई शारीरिक रूप से घायल हुआ? किसी को चिकित्सा सहायता की जरूरत पड़ी?",
        "mr": "कोणी शारीरिकदृष्ट्या जखमी झाले का? कोणाला वैद्यकीय मदत लागली का?",
        "ta": "யாரேனும் உடல் ரீதியாக காயமடைந்தார்களா?",
    },
    "probe_assault_mlc": {
        "hi": "क्या आप अस्पताल या डॉक्टर के पास गए? क्या Medico-Legal Certificate (MLC) है?",
        "mr": "तुम्ही रुग्णालय किंवा डॉक्टरकडे गेलात का? MLC आहे का?",
        "ta": "மருத்துவமனை சென்றீர்களா? MLC உள்ளதா?",
    },
    "probe_cyber_amount": {
        "hi": "कितना पैसा खोया? (₹ में)\nकौन सा प्लेटफॉर्म? (UPI / बैंक / वेबसाइट / ऐप)",
        "mr": "किती पैसे गेले? (₹ मध्ये)\nकोणते प्लॅटफॉर्म? (UPI / बँक / वेबसाइट / ऐप)",
        "ta": "எவ்வளவு பணம் இழந்தீர்கள்? (₹)\nஎந்த தளம்? (UPI / வங்கி / வலைதளம் / ஆப்)",
    },
    "probe_harassment_online": {
        "hi": "क्या यह उत्पीड़न ऑनलाइन हुआ, व्यक्तिगत रूप से, या दोनों?",
        "mr": "हा छळ ऑनलाइन झाला, प्रत्यक्ष, की दोन्ही?",
        "ta": "இந்த தொல்லை ஆன்லைனில் நடந்ததா, நேரடியாகவா, இல்லை இரண்டுமா?",
    },
    "probe_accused": {
        "hi": "क्या आप जानते हैं कि यह किसने किया? (नाम, उम्र, पहचान, या संबंध — या 'पता नहीं' लिखें)",
        "mr": "हे कोणी केले हे तुम्हाला माहीत आहे का? (नाव, वय, वर्णन — किंवा 'माहीत नाही' लिहा)",
        "ta": "யார் இதை செய்தார்கள் என்று தெரியுமா? (பெயர், வயது — அல்லது 'தெரியாது' சொல்லுங்கள்)",
    },
    "probe_witnesses": {
        "hi": "क्या कोई गवाह थे? (नाम और संपर्क नंबर यदि उपलब्ध हो)",
        "mr": "काही साक्षीदार होते का? (नाव आणि संपर्क क्रमांक उपलब्ध असल्यास)",
        "ta": "சாட்சிகள் இருந்தார்களா? (பெயர்கள் மற்றும் தொடர்பு எண்கள் கிடைத்தால்)",
    },
    "probe_witnesses_detail": {
        "hi": "गवाह मौजूद थे — क्या आप उनके नाम या संपर्क नंबर बता सकते हैं?",
        "mr": "साक्षीदार उपस्थित होते — त्यांची नावे किंवा संपर्क क्रमांक सांगता येईल का?",
        "ta": "சாட்சிகள் இருந்தார்கள் — அவர்களின் பெயர்கள் அல்லது தொடர்பு எண்கள் சொல்ல முடியுமா?",
    },
    "probe_evidence": {
        "hi": "क्या संलग्न करने के लिए सबूत है? (फ़ोटो, वीडियो, स्क्रीनशॉट, रिपोर्ट — 10 फ़ाइलें तक)",
        "mr": "संलग्न करण्यासाठी काही पुरावे आहेत का? (फोटो, व्हिडिओ, अहवाल — 10 फाइल्सपर्यंत)",
        "ta": "இணைக்க சான்றுகள் உள்ளதா? (புகைப்படங்கள், வீடியோக்கள் — 10 கோப்புகள் வரை)",
    },
    "probe_informant_name": {
        "hi": "लगभग हो गया! FIR के लिए आपका पूरा नाम (जैसा शिकायत पर दिखेगा):",
        "mr": "जवळजवळ झाले! FIR साठी तुमचे पूर्ण नाव (तक्रारीवर जसे दिसेल):",
        "ta": "கிட்டத்தட்ட முடிந்தது! FIR க்கு உங்கள் முழு பெயர்:",
    },
    "probe_informant_address": {
        "hi": "आपका पूरा पता (घर/फ्लैट नंबर, सड़क, क्षेत्र, शहर, PIN कोड):",
        "mr": "तुमचा पूर्ण पत्ता (घर/फ्लॅट नंबर, रस्ता, परिसर, शहर, PIN कोड):",
        "ta": "உங்கள் முழு முகவரி (வீடு/அடுக்குமாடி எண், தெரு, நகரம், PIN குறியீடு):",
    },
    "probe_informant_phone": {
        "hi": "आपका मोबाइल नंबर (पुलिस इस पर संपर्क करेगी):",
        "mr": "तुमचा मोबाईल नंबर (पोलीस याद्वारे संपर्क साधतील):",
        "ta": "உங்கள் மொபைல் எண் (காவல்துறை தொடர்பு கொள்ள):",
    },
    "probe_informant_phone_retry": {
        "hi": "वह वैध मोबाइल नंबर नहीं लगता। कृपया 10 अंकों का भारतीय नंबर दर्ज करें:",
        "mr": "ते वैध मोबाईल नंबर नाही. 10 अंकी भारतीय मोबाईल नंबर टाका:",
        "ta": "சரியான மொபைல் எண் போல் தெரியவில்லை. 10 இலக்க இந்திய மொபைல் எண் உள்ளிடுங்கள்:",
    },
    "probe_force_used": {
        "hi": "क्या कोई बल, धमकी या हथियार का उपयोग हुआ?",
        "mr": "काही बळाचा वापर, धमकी किंवा शस्त्र वापरले का?",
        "ta": "வலிந்து கொடுமைப்படுத்தல், மிரட்டல் அல்லது ஆயுதம் பயன்பாடு இருந்ததா?",
    },
    "probe_stolen_phone_imei": {
        "hi": "क्या फोन चोरी हुआ? IMEI नंबर पता है? (*#06# डायल करें)",
        "mr": "फोन चोरी झाला का? IMEI नंबर माहीत आहे का? (*#06# डायल करा)",
        "ta": "தொலைபேசி திருடப்பட்டதா? IMEI எண் தெரியுமா? (*#06# அழையுங்கள்)",
    },
    "probe_sim_blocked": {
        "hi": "क्या SIM ब्लॉक किया? (Airtel 121 / Jio 198 / BSNL 1500 पर कॉल करें)",
        "mr": "SIM ब्लॉक केले का? (Airtel 121 / Jio 198 / BSNL 1500 वर कॉल करा)",
        "ta": "SIM தடுத்தீர்களா? (Airtel 121 / Jio 198 / BSNL 1500)",
    },
    "probe_incident_place_detail": {
        "hi": "कोई अतिरिक्त स्थान विवरण? (बस नंबर, ट्रेन डिब्बा, वाहन पंजीकरण आदि)",
        "mr": "कोणतेही अतिरिक्त ठिकाण तपशील? (बस क्रमांक, ट्रेन डबा, वाहन नोंदणी)",
        "ta": "கூடுதல் இட விவரங்கள்? (பேருந்து எண், ரயில் பெட்டி, வாகன எண்)",
    },
    "probe_transaction_ids": {
        "hi": "Transaction ID(s) या UTR नंबर बताएं। (SMS, ईमेल या ऐप इतिहास में देखें)",
        "mr": "Transaction ID(s) किंवा UTR क्रमांक द्या. (SMS, ईमेल किंवा ऐप इतिहासात पहा)",
        "ta": "Transaction ID(s) அல்லது UTR எண்களைப் பகிர்ந்துகொள்ளுங்கள்.",
    },
    "probe_scammer_contact": {
        "hi": "ठग के संपर्क विवरण हैं? (फोन, ईमेल, UPI ID, वेबसाइट — जो भी हो)",
        "mr": "फसवणूक करणाऱ्याचे संपर्क तपशील आहेत का? (फोन, ईमेल, UPI ID)",
        "ta": "மோசடி செய்தவரின் தொடர்பு விவரங்கள் உள்ளதா? (தொலைபேசி, மின்னஞ்சல், UPI ID)",
    },
    "probe_dv_duration": {
        "hi": "यह कब से हो रहा है? (उदा: 3 महीने, 2 साल)",
        "mr": "हे किती दिवसांपासून होत आहे? (उदा: 3 महिने, 2 वर्षे)",
        "ta": "இது எவ்வளவு காலமாக நடக்கிறது? (உதாரணம்: 3 மாதங்கள், 2 ஆண்டுகள்)",
    },
    "probe_dv_children": {
        "hi": "क्या घर में 18 वर्ष से कम आयु के प्रभावित बच्चे हैं?",
        "mr": "घरात 18 वर्षांपेक्षा कमी वयाची प्रभावित मुले आहेत का?",
        "ta": "வீட்டில் பாதிக்கப்பட்ட 18 வயதுக்குட்பட்ட குழந்தைகள் உள்ளார்களா?",
    },
    "probe_posh_employer": {
        "hi": "नियोक्ता/कंपनी का नाम और कार्यस्थल का पता?",
        "mr": "नियोक्त्याचे/कंपनीचे नाव आणि कार्यस्थळाचा पत्ता?",
        "ta": "நிறுவனத்தின் பெயர் மற்றும் பணியிட முகவரி?",
    },
    "probe_posh_role": {
        "hi": "आपकी भूमिका और उत्पीड़क की पदवी?",
        "mr": "तुमची भूमिका आणि छळ करणाऱ्याचे पद?",
        "ta": "உங்கள் பதவி மற்றும் தொல்லை செய்தவரின் பதவி?",
    },
    "probe_posh_icc": {
        "hi": "क्या ICC में पहले से रिपोर्ट किया गया है?",
        "mr": "ICC ला आधी तक्रार केली आहे का?",
        "ta": "ICC யிடம் ஏற்கனவே புகாரளித்தீர்களா?",
    },
    "probe_consumer_company": {
        "hi": "शामिल कंपनी या प्लेटफॉर्म का नाम?",
        "mr": "सहभागी कंपनी किंवा प्लॅटफॉर्मचे नाव?",
        "ta": "சம்பந்தப்பட்ட நிறுவனம் அல்லது தளம்?",
    },
    "probe_consumer_order_id": {
        "hi": "ऑर्डर ID, Transaction संदर्भ, या शिकायत नंबर?",
        "mr": "ऑर्डर ID, व्यवहार संदर्भ, किंवा तक्रार क्रमांक?",
        "ta": "ஆர்டர் ID, பரிவர்த்தனை எண், அல்லது புகார் எண் உள்ளதா?",
    },
}


def _tm(key: str, lang: str, **kwargs) -> str:
    """Return translated message for `key` in `lang`, falling back to English."""
    trans = _STATIC_TRANS.get(key, {})
    msg = trans.get(lang) or trans.get("en", f"[{key}]")
    if kwargs:
        try:
            msg = msg.format(**kwargs)
        except (KeyError, IndexError):
            pass
    return msg


def _tp(probe_key: str, lang: str) -> str:
    """Return translated probe question for `probe_key` in `lang`, falling back to PROBE_Q message."""
    trans = _PROBE_MSG_TRANS.get(probe_key, {})
    return trans.get(lang) or PROBE_Q.get(probe_key, {}).get("message", "")


# ─── Stage constants ───────────────────────────────────────────────────────────
STAGE_SAFETY_GATE     = "safety_gate"
STAGE_FREE_NARRATIVE  = "free_narrative"
STAGE_PROBE           = "probe"
STAGE_GPS_CONFIRM     = "gps_confirm"
STAGE_SECTION_SUGGEST = "section_suggest"
STAGE_READ_BACK       = "read_back"
STAGE_CORRECTION      = "correction"
STAGE_DRAFT           = "draft"
STAGE_COMPLETED       = "completed"

# ─── Input type hints (controls frontend rendering) ───────────────────────────
INPUT_TEXT        = "text"
INPUT_VOICE_TEXT  = "voice_or_text"
INPUT_QUICK_REPLY = "quick_reply"
INPUT_GPS         = "gps"
INPUT_EVIDENCE    = "evidence"
INPUT_CONFIRM     = "confirm"
INPUT_DONE        = "done"

# ─── Safety flag detection ────────────────────────────────────────────────────
_SAFETY_KW: dict[str, list[str]] = {
    "DV":     ["domestic violence", "husband beat", "dahej", "gharelu hinsa",
               "pati ne", "dowry", "marital violence", "wife beating"],
    "SEXUAL": ["rape", "sexual assault", "molest", "balaatkaar", "sexual harassment",
               "groped", "eve teas", "outrage modesty"],
    "POCSO":  ["child abuse", "minor abused", "pocso", "child rape", "childline"],
}

# Keywords indicating the user is in IMMEDIATE danger right now
_EMERGENCY_NARRATIVE_KW = [
    # Direct calls for help
    "help me now", "please help me", "please come help", "need help now",
    "need help right now", "right now please", "please come",
    # Attacker presence
    "he is here", "he's here", "she is here", "she's here",
    "they are here", "he has come", "she has come",
    # Being attacked RIGHT NOW
    "being attacked", "attacking me", "beating me now", "is beating me",
    "hurting me right now", "harming me now",
    # Lethal threat NOW
    "will kill me", "kill me now", "kill me right now", "trying to kill me",
    "going to kill me", "wants to kill me", "killing me",
    # Danger / confinement
    "in danger now", "in danger right now", "right now in danger",
    "save me", "can't escape", "trapped here", "cannot escape",
    # Following/stalking NOW
    "someone following me", "following me right now", "being followed now",
    # Distress phrases
    "please call police", "running away", "he is outside",
]


def _detect_safety_flags(text: str) -> list[str]:
    t = text.lower()
    return [f for f, kws in _SAFETY_KW.items() if any(k in t for k in kws)]


def _detect_immediate_danger(text: str) -> bool:
    """Return True if narrative suggests user is in danger RIGHT NOW."""
    t = text.lower()
    return any(kw in t for kw in _EMERGENCY_NARRATIVE_KW)


# Keywords for relative date detection
_RELATIVE_DATE_KW = {
    "today": ["today", "aaj", "this morning", "this evening", "this afternoon",
              "just now", "a few hours ago", "tonight"],
    "yesterday": ["yesterday", "kal", "last night", "last evening"],
    "last_week": ["last week", "pichle hafte", "a week ago"],
}


# ─── Helper validators ────────────────────────────────────────────────────────

# Location-type words that suggest a valid place answer
_PLACE_KW = [
    "road", "street", "nagar", "colony", "area", "sector", "phase", "block",
    "city", "town", "village", "district", "state", "tehsil", "taluk",
    "park", "market", "market", "station", "chowk", "bazaar", "marg", "lane",
    "near", "beside", "opposite", "behind", "in front", "next to",
    "at", "in the", "on the", "under", "inside", "outside", "between",
    "mall", "hospital", "school", "college", "office", "building", "flat",
    "house", "floor", "floor", "delhi", "mumbai", "bangalore", "bengaluru",
    "chennai", "hyderabad", "kolkata", "pune", "ahmedabad", "surat",
    "jaipur", "lucknow", "kanpur", "nagpur", "patna", "indore", "bhopal",
    "pin", "pincode", "zip", "highway", "nh-", "sh-", "bridge",
]

# Patterns that clearly indicate a non-place answer
_NON_PLACE_PATTERNS = [
    r"^how (do|can|will|should|would|to)",
    r"^what (is|are|was|were|should|would|can)",
    r"^why (is|are|was|did|should|would)",
    r"^(can|could|will|would|should|shall) (you|i|we|he|she|they)",
    r"^please (tell|explain|help|advise|suggest|let)",
    r"^i (want|need|don't|dont|wish|am|was|have|had)",
    r"^(not|na|none|nil|skip)",
]


def _is_valid_place_response(text: str) -> bool:
    """Return True if text looks like a place description."""
    if not text or len(text.strip()) < 3:
        return True  # accept very short answers (abbreviations etc.)
    t = text.lower().strip()
    # Reject clear non-place patterns (questions, requests, denials)
    for pat in _NON_PLACE_PATTERNS:
        if re.match(pat, t):
            return False
    # Accept if it contains any place keyword
    if any(kw in t for kw in _PLACE_KW):
        return True
    # Accept short answers (could be neighbourhood name, place abbreviation)
    if len(t.split()) <= 4:
        return True
    # Long sentence with no location keywords = likely off-topic
    return False


_WITNESS_KW = ["witness", "saw", "they were", "my friend", "present", "there"]
_VAGUE_WITNESS = ["yes", "yeah", "yep", "sure", "there were", "yes there",
                  "they were witnesses", "they are witnesses", "they were present",
                  "my friends were", "few people"]

# Common sentence-starter or generic words that are capitalised but are NOT person names
_COMMON_CAPS: set[str] = {
    "Yes", "No", "There", "They", "The", "This", "That", "Some", "My",
    "His", "Her", "Our", "We", "He", "She", "It", "But", "And", "Or",
    "Few", "Many", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight",
    "People", "Person", "Witnesses", "Witness", "Friends", "Neighbours",
    "Neighbors", "Bystanders", "Someone", "Anyone", "Everyone",
    "About", "Around", "Nearby", "Present",
}


def _is_witness_answer_in_accused_probe(text: str) -> bool:
    """Return True if an answer to 'who did this' is actually about witnesses."""
    t = text.lower().strip()
    return any(kw in t for kw in ["witness", "witnesses", "they were there",
                                   "they witnessed", "saw it", "they saw"])


def _is_vague_witness_response(text: str) -> bool:
    """Return True if witness answer is too vague (no names/numbers given).

    A witness answer is considered specific if it contains:
    - at least one digit run that looks like a phone number, OR
    - at least one capitalized word that is NOT a common English word
      (i.e., a real person name such as 'Ramesh', 'Priya', 'Suresh')
    """
    if not text:
        return True
    t = text.lower().strip()
    # Very short or just "yes / yeah / sure"
    if len(t) < 30 and any(t.startswith(v) for v in _VAGUE_WITNESS):
        return True
    # Check for any digit sequence (phone numbers)
    has_digit = any(c.isdigit() for c in text)
    if has_digit:
        return False
    # Check for a real person name: capitalised word NOT in our common-caps exclusion list
    cap_words = re.findall(r"\b[A-Z][a-z]{2,}\b", text)
    has_person_name = any(w not in _COMMON_CAPS for w in cap_words)
    if has_person_name:
        return False
    # No phone number, no person name → vague
    return True


_PLACEHOLDER_PHONE = [
    r"^0+$",          # 000, 00000, 0000000000
    r"^1+$",          # 111...
    r"^9+$",
    r"^\d{1,4}$",     # too short (1–4 digits)
    r"^(n/?a|na|none|no|not available|unknown|skip)$",
    r"put as",        # "put as 00000"
    r"xxx+",
]


def _date_validation_error(text: str) -> Optional[str]:
    """
    DHARA-B009 / S1-5: Reject impossible calendar dates (e.g. 32/13/2026) and
    future dates (an incident can't be reported before it happens) before
    they reach a draft. Only flags a structured D/M/Y-style numeric pattern —
    free narrative ("last Monday", "Diwali week") has no such pattern and is
    accepted as-is for the LLM extractor to interpret.
    Returns None if the date is valid, else a user-facing error message.
    """
    m = re.search(r'\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})\b', text.strip())
    if not m:
        return None
    d, mo, y = m.groups()
    try:
        d_i, mo_i, y_i = int(d), int(mo), int(y)
        y_i = y_i + 2000 if y_i < 100 else y_i
        parsed = date(y_i, mo_i, d_i)
    except ValueError:
        return "That date doesn't look valid — please check the day and month (e.g. 15/03/2026)."
    if parsed > date.today():
        return "That date is in the future — the incident date can't be after today. Please re-check (e.g. 15/03/2026)."
    return None


def _validate_date_input(text: str) -> bool:
    """Backward-compatible boolean wrapper around _date_validation_error."""
    return _date_validation_error(text) is None


def _validate_time_input(text: str) -> bool:
    """DHARA-B009: Reject impossible clock times (e.g. 25:90)."""
    m = re.search(r'\b(\d{1,2})[:.](\d{2})\b', text.strip())
    if not m:
        return True
    h, mi = int(m.group(1)), int(m.group(2))
    return h <= 23 and mi <= 59


def _is_valid_phone(text: str) -> bool:
    """Return True if text looks like a real contact number."""
    if not text:
        return False
    t = text.strip().lower()
    # Match placeholder patterns
    for pat in _PLACEHOLDER_PHONE:
        if re.search(pat, t):
            return False
    # Extract digits only
    digits = re.sub(r"\D", "", text)
    # Indian mobile: 10 digits, starts with 6-9
    if len(digits) == 10 and digits[0] in "6789":
        return True
    # Landline with STD (8-11 digits including STD code)
    if 8 <= len(digits) <= 11:
        return True
    return False


# ─── Date contradiction helpers ───────────────────────────────────────────────

_TODAY_KW = ["today", "aaj", "this evening", "this morning", "this afternoon",
             "just now", "a few hours", "tonight", "aaj raat", "aaj subah"]
_YESTERDAY_KW = ["yesterday", "kal", "last night", "last evening"]


def _narrative_mentions_today(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in _TODAY_KW)


def _narrative_mentions_yesterday(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in _YESTERDAY_KW)


def _has_date_conflict(narrative: str, extracted_date: Optional[str]) -> Optional[str]:
    """
    Returns the conflict message if there's a date contradiction, else None.
    Checks:
    - Narrative says "today" but extracted date != today
    - Narrative says "yesterday" but extracted date != yesterday
    - Extracted year differs significantly from current year
    """
    if not extracted_date:
        return None
    today = date.today()
    today_str = today.isoformat()  # YYYY-MM-DD
    today_fmt = today.strftime("%d %B %Y")

    # Try to parse the extracted date
    parsed = None
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            parsed = datetime.strptime(extracted_date.strip(), fmt).date()
            break
        except ValueError:
            continue

    if parsed is None:
        return None

    # Check 1: narrative says "today" but extracted date is not today
    if _narrative_mentions_today(narrative) and parsed != today:
        # Format the extracted date nicely
        parsed_fmt = parsed.strftime("%d %B %Y")
        return (
            f"You mentioned 'today' — did this happen on **{today_fmt}** (today) "
            f"or was it **{parsed_fmt}** as mentioned?"
        )

    # Check 2: narrative says "yesterday" but extracted date is not yesterday
    from datetime import timedelta
    yesterday = today - timedelta(days=1)
    if _narrative_mentions_yesterday(narrative) and parsed != yesterday:
        parsed_fmt = parsed.strftime("%d %B %Y")
        yesterday_fmt = yesterday.strftime("%d %B %Y")
        return (
            f"You mentioned 'yesterday' — did this happen on **{yesterday_fmt}** "
            f"or was it **{parsed_fmt}** as mentioned?"
        )

    # Check 3: extracted date is in the future (S1-5 — an incident can't be
    # reported before it happens)
    if parsed > today:
        parsed_fmt = parsed.strftime("%d %B %Y")
        return (
            f"I noted the date as {parsed_fmt}, which is in the future — "
            f"could you confirm the correct date? (Today is {today_fmt})"
        )

    # Check 4: extracted year is more than 1 year before today (likely LLM default)
    if abs(parsed.year - today.year) > 1:
        parsed_fmt = parsed.strftime("%d %B %Y")
        return (
            f"I noted the date as {parsed_fmt} — could you confirm the year? "
            f"(The current year is {today.year})"
        )

    return None


# ─── Incident type keyword fallback classifier ────────────────────────────────

_INCIDENT_KW_MAP: dict[str, list[str]] = {
    "theft":     ["stolen", "steal", "stole", "theft", "chori", "pickpocket",
                  "missing phone", "missing wallet", "missing purse",
                  "grabbed my", "grabbed the", "snatched my", "took my phone",
                  "took my wallet", "took my bag", "took my laptop", "took my car",
                  "my phone was taken", "mobile stolen", "bike stolen", "car stolen"],
    "robbery":   ["robbery", "looted", "loot", "robbed",
                  "snatched at", "knife", "knifepoint", "gunpoint", "at gunpoint",
                  "at knifepoint", "threatened with", "threatened at"],
    "assault":   ["beat", "beaten", "hit me", "hit us", "punch", "kick", "slap",
                  "physical", "attacked", "attack", "assault", "maara", "pita",
                  "injury", "hospital", "fracture", "wound"],
    "harassment": ["harass", "bully", "bullied", "making fun", "mock",
                   "teas", "taunt", "verbal abuse", "insulted",
                   "humiliate", "intimidat", "menac", "mischief", "nuisance",
                   "eve teas", "ragging"],
    "cyber_fraud":["online", "fraud", "cyber", "upi", "payment", "bank",
                   "account", "otp", "phishing", "scam", "cheated online",
                   "fake website", "fake call", "whatsapp"],
    "domestic_violence": ["husband", "wife", "spouse", "domestic", "dowry",
                          "marital", "in-law", "inlaw", "gharelu", "498a", "cruelty"],
    "sexual_harassment": ["molest", "grope", "voyeur", "stalk", "follow me",
                          "sexual", "inappropriate touch", "eve teas"],
    "posh":      ["posh", "workplace harassment", "office harassment", "boss",
                  "manager harass", "colleague harass", "sexual harassment at work",
                  "internal complaints committee", "icc complaint"],
    "consumer_fraud": ["amazon", "flipkart", "myntra", "meesho", "consumer",
                       "e-commerce", "online shopping", "refund not received",
                       "defective product", "delivery fraud", "fake product",
                       "never delivered", "wrong product", "consumer court"],
    "murder":    ["killed", "death", "murder", "dead", "die", "hataaya"],
}

# ─── Module-specific welcome messages (shown when user pre-selects a module) ──
_MODULE_WELCOME: dict[str, dict[str, str]] = {
    "cybercrime": {
        "en": "🔐 I'll help you file a Cybercrime complaint.\n\nThis covers: online fraud, UPI/banking scams, hacking, social media abuse, and sextortion.\n\n📞 Report at cybercrime.gov.in or call Cybercrime Helpline 1930 first to preserve evidence.",
        "hi": "🔐 मैं आपकी साइबर अपराध की शिकायत दर्ज करने में मदद करूंगा।\n\nइसमें शामिल है: ऑनलाइन धोखाधड़ी, UPI/बैंकिंग घोटाले, हैकिंग, सोशल मीडिया दुरुपयोग।\n\n📞 पहले Cybercrime Helpline 1930 पर कॉल करें।",
        "mr": "🔐 मी तुम्हाला सायबर गुन्ह्याची तक्रार दाखल करण्यात मदत करेन।\n\nयामध्ये समाविष्ट आहे: ऑनलाइन फसवणूक, UPI/बँकिंग घोटाळे.\n\n📞 प्रथम Cybercrime Helpline 1930 वर कॉल करा.",
        "ta": "🔐 சைபர் கிரைம் புகார் பதிவு செய்ய உங்களுக்கு உதவுகிறேன்.\n\n📞 முதலில் Cybercrime Helpline 1930 ஐ அழையுங்கள்.",
    },
    "domestic_violence": {
        "en": "🏠 I'll help you file a complaint under the Protection of Women from Domestic Violence Act and BNS Section 85 (cruelty).\n\n📞 Women Helpline: 181 (24×7 FREE)\n📞 NCW: 7827170170",
        "hi": "🏠 मैं घरेलू हिंसा और BNS धारा 85 के तहत शिकायत दर्ज करने में मदद करूंगा।\n\n📞 महिला हेल्पलाइन: 181 (24×7 मुफ्त)",
        "mr": "🏠 मी घरगुती हिंसाचार कायदा आणि BNS कलम 85 अंतर्गत तक्रार दाखल करण्यात मदत करेन.\n\n📞 महिला हेल्पलाइन: 181",
        "ta": "🏠 குடும்ப வன்முறை சட்டம் மற்றும் BNS பிரிவு 85 கீழ் புகார் பதிவு செய்ய உதவுகிறேன்.\n\n📞 மகளிர் உதவி எண்: 181",
    },
    "theft": {
        "en": "📱 I'll help you file a Theft / Robbery / Burglary complaint under BNS Sections 303–310.\n\nTip: Note your device's IMEI (dial *#06#) — it helps police trace stolen phones.",
        "hi": "📱 मैं BNS धारा 303-310 के तहत चोरी / डकैती / सेंधमारी की शिकायत दर्ज करने में मदद करूंगा।\n\nटिप: अपने डिवाइस का IMEI नोट करें (*#06# डायल करें)।",
        "mr": "📱 मी BNS कलम 303-310 अंतर्गत चोरी / दरोडा / घरफोडीची तक्रार दाखल करण्यात मदत करेन.",
        "ta": "📱 BNS பிரிவு 303-310 கீழ் திருட்டு / கொள்ளை புகார் பதிவு செய்ய உதவுகிறேன்.",
    },
    "posh": {
        "en": "👔 I'll help you file a POSH complaint (Workplace Sexual Harassment) under BNS Section 74, 75 and the POSH Act, 2013.\n\n⚖️ Your employer is legally required to have an Internal Complaints Committee (ICC). You may file there first, OR directly with the police.",
        "hi": "👔 मैं BNS धारा 74, 75 और POSH अधिनियम 2013 के तहत कार्यस्थल यौन उत्पीड़न की शिकायत दर्ज करने में मदद करूंगा।",
        "mr": "👔 मी BNS कलम 74, 75 आणि POSH कायदा 2013 अंतर्गत कामाच्या ठिकाणी लैंगिक छळाची तक्रार दाखल करण्यात मदत करेन.",
        "ta": "👔 BNS பிரிவு 74, 75 மற்றும் POSH சட்டம் 2013 கீழ் பணியிட பாலியல் தொல்லை புகார் பதிவு செய்ய உதவுகிறேன்.",
    },
    "consumer_fraud": {
        "en": "💳 I'll help you file a Consumer / Banking Fraud complaint under BNS Section 318 (cheating) and the Consumer Protection Act, 2019.\n\n📞 National Consumer Helpline: 1915\n🌐 consumerhelpline.gov.in",
        "hi": "💳 मैं BNS धारा 318 (धोखाधड़ी) और उपभोक्ता संरक्षण अधिनियम 2019 के तहत शिकायत दर्ज करने में मदद करूंगा।\n\n📞 राष्ट्रीय उपभोक्ता हेल्पलाइन: 1915",
        "mr": "💳 मी BNS कलम 318 आणि ग्राहक संरक्षण कायदा 2019 अंतर्गत तक्रार दाखल करण्यात मदत करेन.\n\n📞 राष्ट्रीय ग्राहक हेल्पलाइन: 1915",
        "ta": "💳 BNS பிரிவு 318 மற்றும் நுகர்வோர் பாதுகாப்பு சட்டம் 2019 கீழ் புகார் பதிவு செய்ய உதவுகிறேன்.\n\n📞 தேசிய நுகர்வோர் உதவி எண்: 1915",
    },
}

# ─── BNS section heading translations ────────────────────────────────────────
_BNS_HEADING_TRANS: dict[str, dict[str, str]] = {
    "74":  {"hi": "महिला की लज्जा भंग", "mr": "स्त्रीची लज्जाभंग", "ta": "பெண்ணின் கற்பை அழிக்கும் முயற்சி", "te": "మహిళ మర్యాద హాని", "kn": "ಮಹಿಳೆ ಮರ್ಯಾದೆ ಭಂಗ"},
    "75":  {"hi": "यौन उत्पीड़न", "mr": "लैंगिक छळ", "ta": "பாலியல் தொல்லை", "te": "లైంగిక వేధింపు", "kn": "ಲೈಂಗಿಕ ಕಿರುಕುಳ"},
    "77":  {"hi": "दृश्यरतिकता", "mr": "अश्लील छायाचित्रण", "ta": "ஒட்டுவேடிக்கை", "te": "అసభ్య చిత్రీకరణ", "kn": "ಅಶ್ಲೀಲ ಚಿತ್ರೀಕರಣ"},
    "78":  {"hi": "पीछा करना", "mr": "पाठलाग करणे", "ta": "பின்தொடர்தல்", "te": "వెంబడించడం", "kn": "ಹಿಂಬಾಲಿಸುವಿಕೆ"},
    "85":  {"hi": "पति या रिश्तेदारों द्वारा क्रूरता", "mr": "पती किंवा नातेवाईकांकडून क्रौर्य", "ta": "கணவன் அல்லது உறவினர்களால் கொடுமை", "te": "భర్త లేదా బంధువుల క్రూరత్వం", "kn": "ಪತಿ ಅಥವಾ ಸಂಬಂಧಿಕರಿಂದ ಕ್ರೌರ್ಯ"},
    "101": {"hi": "हत्या", "mr": "खून", "ta": "கொலை", "te": "హత్య", "kn": "ಕೊಲೆ"},
    "109": {"hi": "हत्या का प्रयास", "mr": "खुनाचा प्रयत्न", "ta": "கொலை முயற்சி", "te": "హత్యా ప్రయత్నం", "kn": "ಕೊಲೆ ಪ್ರಯತ್ನ"},
    "115": {"hi": "स्वैच्छिक चोट", "mr": "ऐच्छिक दुखापत", "ta": "விருப்பமான காயம்", "te": "స్వచ్ఛంద గాయం", "kn": "ಸ್ವಯಂಪ್ರೇರಿತ ಗಾಯ"},
    "117": {"hi": "स्वैच्छिक चोट का प्रयास", "mr": "ऐच्छिक दुखापत करण्याचा प्रयत्न", "ta": "காயப்படுத்தும் முயற்சி"},
    "303": {"hi": "चोरी", "mr": "चोरी", "ta": "திருட்டு", "te": "దొంగతనం", "kn": "ಕಳ್ಳತನ"},
    "304": {"hi": "घर में चोरी", "mr": "घरफोडी", "ta": "வீட்டு திருட்டு", "te": "ఇంట్లో దొంగతనం", "kn": "ಮನೆ ಕಳ್ಳತನ"},
    "309": {"hi": "लूट", "mr": "दरोडा", "ta": "கொள்ளை", "te": "దోపిడీ", "kn": "ದರೋಡೆ"},
    "310": {"hi": "डकैती", "mr": "जमावाने दरोडा", "ta": "கும்பல் கொள்ளை", "te": "దొంగల ముఠా దోపిడీ", "kn": "ಡಕಾಯಿತಿ"},
    "316": {"hi": "आपराधिक विश्वासघात", "mr": "फौजदारी विश्वासघात", "ta": "நம்பிக்கை துரோகம்", "te": "నమ్మకద్రోహం", "kn": "ನಂಬಿಕೆ ದ್ರೋಹ"},
    "318": {"hi": "धोखाधड़ी", "mr": "फसवणूक", "ta": "மோசடி", "te": "మోసం", "kn": "ವಂಚನೆ"},
    "351": {"hi": "आपराधिक धमकी", "mr": "फौजदारी धमकी", "ta": "குற்றவியல் மிரட்டல்", "te": "నేర బెదిరింపు", "kn": "ಕ್ರಿಮಿನಲ್ ಬೆದರಿಕೆ"},
    "352": {"hi": "जानबूझकर अपमान", "mr": "जाणूनबुजून अपमान", "ta": "வேண்டுமென்றே அவமானம்", "te": "ఉద్దేశపూర్వక అవమానం", "kn": "ಉದ್ದೇಶಪೂರ್ವಕ ಅವಮಾನ"},
}


def _translate_bns_heading(section_number: str, heading: str, language: str) -> str:
    """Return translated BNS section heading if available, else original English."""
    if language in ("en", "english", ""):
        return heading
    trans = _BNS_HEADING_TRANS.get(str(section_number), {})
    return trans.get(language, heading)


# DHARA-P0-1: generic words that must not ADD an incident type on top of the
# LLM's classification (e.g. "no injury, no hospital" was adding assault,
# "paid by UPI" was adding cyber fraud, "my husband" was adding domestic
# violence). They still count when the LLM returns nothing.
_KW_WEAK: set[str] = {
    "injury", "hospital", "physical", "attack", "wound",
    "online", "bank", "account", "payment", "whatsapp", "upi", "otp",
    "husband", "wife", "spouse", "boss", "follow me",
}
_KW_NEG_BEFORE = ("no", "not", "without", "never", "didn't", "did not",
                  "wasn't", "was not", "nobody", "no one", "nahi", "nahin")
_KW_NEG_AFTER = ("nahi", "nahin", "नहीं", "नही", "नाही", "இல்லை")


def _kw_negated(t: str, i: int, kw: str) -> bool:
    """True when the keyword at index i is negated within its own clause."""
    before = t[max(0, i - 30):i]
    for sep in (".", ",", ";", "!", "?", " but ", " and ", "\n"):
        if sep in before:
            before = before.rsplit(sep, 1)[-1]
    words = re.findall(r"[\w']+", before)
    joined = " ".join(words)
    if any(re.search(r"(^| )" + re.escape(n) + r"( |$)", joined) for n in _KW_NEG_BEFORE):
        return True
    after = t[i + len(kw):i + len(kw) + 20]
    after = re.split(r"[.,;!?]", after)[0]
    return any(n in after for n in _KW_NEG_AFTER)


def _keyword_classify(narrative: str, strong_only: bool = False) -> list[str]:
    """Fast keyword-based incident type classification as a fallback.

    DHARA-P0-1: negation-aware; strong_only skips generic words (used when
    merging on top of an LLM classification).
    """
    t = narrative.lower()
    found: list[str] = []
    for itype, kws in _INCIDENT_KW_MAP.items():
        for kw in kws:
            if strong_only and kw in _KW_WEAK:
                continue
            start, hit = 0, False
            while True:
                i = t.find(kw, start)
                if i < 0:
                    break
                if not _kw_negated(t, i, kw):
                    hit = True
                    break
                start = i + 1
            if hit:
                found.append(itype)
                break
    return found if found else ["other"]


# DHARA-P0-2: yes/no answers in English, Hindi, Marathi and Tamil (typed,
# spoken or tapped). The old startswith("yes") check saved every non-English
# "yes" as "no".
_YES_WORDS = {"yes", "yeah", "yep", "y", "haan", "han", "haa", "ha", "ji",
              "हाँ", "हां", "हा", "जी", "होय", "हो", "ஆம்", "ஆமாம்", "aam", "aama", "hoy"}
_NO_WORDS = {"no", "nope", "n", "nahi", "nahin", "na", "नहीं", "नही", "ना",
             "नाही", "இல்லை", "illai", "illa"}


def _is_affirmative(text: str) -> bool:
    tokens = [w for w in re.split(r"[\s,.!?:;\-—–]+", (text or "").lower().strip()) if w]
    if not tokens:
        return False
    if tokens[0] in _NO_WORDS:
        return False
    if tokens[0] in _YES_WORDS:
        return True
    ts = set(tokens)
    if ts & _NO_WORDS:
        return False
    return bool(ts & _YES_WORDS)


def _is_negative(text: str) -> bool:
    tokens = [w for w in re.split(r"[\s,.!?:;\-—–]+", (text or "").lower().strip()) if w]
    return bool(tokens) and tokens[0] in _NO_WORDS


# ─── Probe question definitions ───────────────────────────────────────────────
PROBE_Q: dict[str, dict] = {
    "probe_date": {
        "message": "When did this happen? (date and year if possible)",
        "slot": "incident_date", "input_type": INPUT_VOICE_TEXT, "skip_label": "Skip",
    },
    "probe_time": {
        "message": "At approximately what time did this happen?",
        "slot": "incident_time", "input_type": INPUT_VOICE_TEXT, "skip_label": "Skip",
    },
    "probe_place_text": {
        "message": "Where did this happen? (area / street / city / state)",
        "slot": "incident_place_text", "input_type": INPUT_VOICE_TEXT,
    },
    "probe_place_gps": {
        "message": "Would you like to pinpoint the exact incident location using GPS?\n(Helps the police identify the exact spot — optional)",
        "slot": "incident_gps", "input_type": INPUT_GPS,
        "skip_label": "No, text description is enough",
    },
    # ── Incident-specific ─────────────────────────────────────────────────────
    "probe_theft_items": {
        "message": "What exactly was stolen? List each item with approximate value.\nE.g., \"iPhone 14 — ₹80,000, wallet with ₹3,000 cash\"",
        "slot": "theft_items", "input_type": INPUT_TEXT, "skip_label": "Skip",
    },
    "probe_injury": {
        "message": "Was anyone physically injured? Did anyone need medical attention?",
        "slot": "injury", "input_type": INPUT_QUICK_REPLY,
        "quick_replies": ["Yes — injuries occurred", "No — no injuries"],
    },
    "probe_assault_mlc": {
        "message": "Did you go to a hospital or doctor? Do you have a Medico-Legal Certificate (MLC)?",
        "slot": "assault_mlc", "input_type": INPUT_TEXT, "skip_label": "Skip",
    },
    "probe_cyber_amount": {
        "message": "How much money was lost? (₹ amount)\nWhat platform was used? (UPI / bank transfer / website / app)",
        "slot": "cyber_amount", "input_type": INPUT_TEXT, "skip_label": "Skip",
    },
    "probe_harassment_online": {
        "message": "Did this harassment happen online, in person, or both?",
        "slot": "harassment_mode", "input_type": INPUT_QUICK_REPLY,
        "quick_replies": ["Online only", "In person only", "Both online and in person"],
    },
    # ── Common optional slots ─────────────────────────────────────────────────
    "probe_accused": {
        "message": "Do you know who did this?\nName, age, appearance, or relationship — or say \"Not known\".",
        "slot": "accused", "input_type": INPUT_TEXT, "skip_label": "Not known",
    },
    "probe_witnesses": {
        "message": "Were there any witnesses?\nNames and contact numbers if available.",
        "slot": "witnesses", "input_type": INPUT_VOICE_TEXT, "skip_label": "No witnesses",
    },
    "probe_witnesses_detail": {
        "message": "You mentioned witnesses were present — could you share their names or contact numbers so the police can reach them?",
        "slot": "witnesses_detail", "input_type": INPUT_VOICE_TEXT,
        "skip_label": "Not available right now",
    },
    "probe_evidence": {
        "message": "Do you have evidence to attach?\n(Photos, videos, screenshots, medical reports, receipts — up to 10 files)",
        "slot": None, "input_type": INPUT_EVIDENCE,
        "skip_label": "No evidence to upload",
    },
    # ── Informant details (always required for the final draft) ───────────────
    "probe_informant_name": {
        "message": "Almost done! I need your personal details for the FIR.\n\nYour full name (as it will appear on the complaint):",
        "slot": "informant_name", "input_type": INPUT_VOICE_TEXT,
    },
    "probe_informant_address": {
        "message": "Your full address (house/flat number, street, area, city, PIN code):",
        "slot": "informant_address", "input_type": INPUT_VOICE_TEXT,
    },
    "probe_informant_phone": {
        "message": "Your mobile number (the police can reach you on this):",
        "slot": "informant_phone", "input_type": INPUT_TEXT,
    },
    "probe_informant_phone_retry": {
        "message": "That doesn't look like a valid mobile number — the police will need to reach you.\nPlease enter a 10-digit Indian mobile number:",
        "slot": "informant_phone", "input_type": INPUT_TEXT, "skip_label": "Skip — I'll share it at the station",
    },
    # ── New incident-specific probes ──────────────────────────────────────────
    "probe_force_used": {
        "message": "Was any force, threat, or weapon used?\n(e.g., pushed, threatened, shown a weapon, snatched by force)",
        "slot": "force_used", "input_type": INPUT_QUICK_REPLY,
        "quick_replies": ["Yes, force was used", "No, no force"],
    },
    "probe_stolen_phone_imei": {
        "message": "Was a phone stolen? If yes, do you know the IMEI number?\n(Dial *#06# or check Settings → About. Helps police trace and block the device)",
        "slot": "stolen_phone_imei", "input_type": INPUT_TEXT,
        "skip_label": "No phone stolen / Don't know IMEI",
    },
    "probe_sim_blocked": {
        "message": "If a SIM card was stolen with the phone, have you blocked it yet?\n(Call your telecom operator's helpline to block: Airtel 121 / Jio 198 / BSNL 1500)",
        "slot": "sim_blocked", "input_type": INPUT_QUICK_REPLY,
        "quick_replies": ["Yes, SIM blocked", "Not yet — I'll do it now", "No SIM was stolen"],
    },
    "probe_incident_place_detail": {
        "message": "Any additional location details?\n(e.g., bus number/route, train compartment, exact seat, vehicle registration)",
        "slot": "incident_place_detail", "input_type": INPUT_TEXT,
        "skip_label": "No additional details",
    },
    "probe_transaction_ids": {
        "message": "Please share the transaction ID(s) or UTR/reference number(s) from your bank or UPI app.\n(Check your SMS, email, or app history — helps police trace the fraud)",
        "slot": "transaction_ids", "input_type": INPUT_TEXT,
        "skip_label": "Don't have them right now",
    },
    "probe_scammer_contact": {
        "message": "Do you have any contact details of the scammer?\n(Phone number, email, UPI ID, website URL, bank account number — share whatever you have)",
        "slot": "scammer_contact", "input_type": INPUT_TEXT,
        "skip_label": "Don't have any contact details",
    },
    # ── Domestic Violence specific ─────────────────────────────────────────────
    "probe_dv_duration": {
        "message": "How long has this been happening?\n(e.g., 3 months, 2 years)",
        "slot": "dv_duration", "input_type": INPUT_TEXT, "skip_label": "Skip",
    },
    "probe_dv_children": {
        "message": "Are there children (under 18) in the household affected by this?\n(You don't need to share their names — this helps the magistrate under the DV Act)",
        "slot": "dv_children", "input_type": INPUT_QUICK_REPLY,
        "quick_replies": ["Yes, children are affected", "No children involved"],
    },
    # ── POSH / Workplace Harassment specific ─────────────────────────────────
    "probe_posh_employer": {
        "message": "What is the name of your employer/company and your workplace address?",
        "slot": "posh_employer", "input_type": INPUT_TEXT, "skip_label": "Skip",
    },
    "probe_posh_role": {
        "message": "What is your role at the company, and what is the harasser's designation or role?",
        "slot": "posh_role", "input_type": INPUT_TEXT, "skip_label": "Skip",
    },
    "probe_posh_icc": {
        "message": "Has this already been reported to your company's Internal Complaints Committee (ICC) under the POSH Act, 2013?",
        "slot": "posh_icc", "input_type": INPUT_QUICK_REPLY,
        "quick_replies": ["Yes — ICC complaint filed", "No — filing directly with police", "No ICC at my workplace"],
    },
    # ── Consumer / Banking Fraud specific ────────────────────────────────────
    "probe_consumer_company": {
        "message": "What is the name of the company or platform involved?\n(e.g., Amazon, Flipkart, HDFC Bank, Paytm, local shop)",
        "slot": "consumer_company", "input_type": INPUT_TEXT, "skip_label": "Skip",
    },
    "probe_consumer_order_id": {
        "message": "Do you have an order ID, transaction reference, or complaint number from the company?",
        "slot": "consumer_order_id", "input_type": INPUT_TEXT,
        "skip_label": "Don't have one",
    },
    "probe_date_confirm": {
        # message is set dynamically when this probe is served
        "message": "Could you confirm the date of the incident?",
        "slot": "incident_date", "input_type": INPUT_QUICK_REPLY,
        "quick_replies": ["Yes, that's correct", "No, let me correct the date"],
        "skip_label": "Keep as is",
    },
}

# ─── Incident type → BNS section mapping (Citation Guard source-of-truth) ─────
# FIX v3.1: BNS 77 (Voyeurism) and 78 (Stalking) removed from general "harassment".
# They are now in dedicated subtypes only.
INCIDENT_SECTIONS: dict[str, list[str]] = {
    "theft":              ["303"],
    "robbery":            ["309"],  # B010: 310 (dacoity) needs a confirmed 5+ person group, never assumed
    "assault":            ["115", "117"],
    "harassment":         ["351", "352"],          # Criminal intimidation + Intentional insult
    "sexual_harassment":  ["74", "75", "77", "78", "351"],  # Sexual offences
    "stalking":           ["78"],
    "cyber_fraud":        ["318"],
    "domestic_violence":  ["85"],
    "posh":               ["74", "75", "78", "351"],  # POSH + Criminal intimidation + Stalking
    "consumer_fraud":     ["318", "316"],             # Cheating + Criminal breach of trust
    "murder":             ["101", "109"],
    "other":              [],
}
# Sections dropped when user confirmed NO injury
_INJURY_REQUIRED: set[str] = {"115", "117", "124"}
# Sections dropped when user confirmed NO property loss
_PROPERTY_REQUIRED: set[str] = {"303", "309", "310"}
# DHARA-B010: Robbery (309) and dacoity (310) require force/threat as a
# material element — never assumed. Dropped when the citizen explicitly
# said no force/threat/weapon was used.
_FORCE_REQUIRED: set[str] = {"309"}


def _build_probe_queue(
    slots: dict, incident_types: list[str],
    date_conflict_msg: Optional[str] = None,
    relative_date_display: Optional[str] = None,
) -> list[str]:
    """Build ordered probe queue from missing slots + incident types."""
    q: list[str] = []
    # Date: conflict, relative confirmation, or ask fresh
    if date_conflict_msg and not slots.get("incident_date"):
        q.append("probe_date")
    elif relative_date_display and slots.get("incident_date") and not slots.get("date_confirmed"):
        q.append("probe_date_confirm")
    elif not slots.get("incident_date"):
        q.append("probe_date")
    if not slots.get("incident_time"):
        q.append("probe_time")
    if not slots.get("incident_place_text"):
        q.append("probe_place_text")
    q.append("probe_place_gps")  # always offered
    # ── Incident-specific ────────────────────────────────────────────────────
    if "theft" in incident_types and not slots.get("theft_items"):
        q.append("probe_theft_items")
    if "theft" in incident_types and slots.get("force_used") is None:
        q.append("probe_force_used")
    if "theft" in incident_types and slots.get("stolen_phone_imei") is None:
        q.append("probe_stolen_phone_imei")
    if "theft" in incident_types and slots.get("sim_blocked") is None:
        q.append("probe_sim_blocked")
    if "theft" in incident_types and slots.get("incident_place_detail") is None:
        q.append("probe_incident_place_detail")
    if any(t in incident_types for t in ("assault", "robbery")):
        if slots.get("injury") is None:
            q.append("probe_injury")
        q.append("probe_assault_mlc")
    if "cyber_fraud" in incident_types and not slots.get("cyber_amount"):
        q.append("probe_cyber_amount")
    if "cyber_fraud" in incident_types and not slots.get("transaction_ids"):
        q.append("probe_transaction_ids")
    if "cyber_fraud" in incident_types and not slots.get("scammer_contact"):
        q.append("probe_scammer_contact")
    if "harassment" in incident_types and slots.get("harassment_mode") is None:
        q.append("probe_harassment_online")
    if "sexual_harassment" in incident_types and slots.get("harassment_mode") is None:
        q.append("probe_harassment_online")
    # ── Domestic Violence specific ────────────────────────────────────────────
    if "domestic_violence" in incident_types:
        if not slots.get("dv_duration"):
            q.append("probe_dv_duration")
        if slots.get("dv_children") is None:
            q.append("probe_dv_children")
    # ── POSH / Workplace Harassment specific ──────────────────────────────────
    if "posh" in incident_types:
        if not slots.get("posh_employer"):
            q.append("probe_posh_employer")
        if not slots.get("posh_role"):
            q.append("probe_posh_role")
        if slots.get("posh_icc") is None:
            q.append("probe_posh_icc")
    # ── Consumer / Banking Fraud specific ─────────────────────────────────────
    if "consumer_fraud" in incident_types:
        if not slots.get("consumer_company"):
            q.append("probe_consumer_company")
        if not slots.get("consumer_order_id"):
            q.append("probe_consumer_order_id")
        if not slots.get("cyber_amount"):
            q.append("probe_cyber_amount")
        if not slots.get("transaction_ids"):
            q.append("probe_transaction_ids")
        if not slots.get("scammer_contact"):
            q.append("probe_scammer_contact")
    # Common optional
    if not slots.get("accused"):
        q.append("probe_accused")
    if not slots.get("witnesses"):
        q.append("probe_witnesses")
    # Evidence (always offered)
    q.append("probe_evidence")
    # Informant details (always required)
    q.extend(["probe_informant_name", "probe_informant_address", "probe_informant_phone"])
    return q


# ─── Nominatim reverse geocoding ─────────────────────────────────────────────
async def reverse_geocode(lat: float, lng: float) -> Optional[str]:
    """Convert GPS coordinates → human-readable address via OpenStreetMap Nominatim."""
    try:
        async with httpx.AsyncClient(timeout=7.0) as client:
            r = await client.get(
                "https://nominatim.openstreetmap.org/reverse",
                params={"format": "json", "lat": lat, "lon": lng},
                headers={"User-Agent": "DHARA-Legal-Aid/1.0 (contact@dhara.app)"},
            )
            if r.status_code == 200:
                data = r.json()
                return data.get("display_name")
    except Exception as e:
        logger.warning(f"[reverse_geocode] failed: {e}")
    return None


# ─── LLM slot extraction ──────────────────────────────────────────────────────
def _make_extract_sys() -> str:
    today_str = date.today().strftime("%d %B %Y")
    return f"""You are a legal intake assistant for Indian FIR complaints.
Extract structured information from an incident narrative.
Return ONLY valid JSON (no markdown, no explanation) with these exact fields:
{{
  "incident_date": null,
  "incident_time": null,
  "incident_place": null,
  "accused": null,
  "witnesses": null,
  "injury": null,
  "property_loss": null,
  "words_or_threats": null,
  "incident_types": [],
  "description": ""
}}
For incident_types use ONLY these exact strings (choose all that apply):
["theft","robbery","assault","harassment","sexual_harassment","stalking","cyber_fraud","domestic_violence","posh","consumer_fraud","murder","other"]
For injury/property_loss: "yes" / "no" / null (null = not clearly mentioned)
DATES: Today's date is {today_str}. When no year is specified in the narrative, assume the current year {date.today().year}.
If the narrative says 'today', set incident_date to today's date: {date.today().isoformat()}.
If the narrative says 'yesterday', set incident_date to {(date.today() - timedelta(days=1)).isoformat()} (yesterday).
"""  # noqa: E501


async def extract_slots_llm(narrative: str, llm_key: str) -> dict:
    """Extract slots from free narrative using Claude Haiku (cost-efficient)."""
    try:
        result = await send_once(
            f"fir-extract-{uuid.uuid4()}",
            _make_extract_sys(),
            f"NARRATIVE:\n{narrative[:2000]}",
            spec=FIR_EXTRACT_SPEC,
            caller="fir_extract_slots",
        )
        raw = (result or "").strip()
        # Strip possible markdown fences
        if raw.startswith("```"):
            raw = re.sub(r"^```[^\n]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw.strip())
        return json.loads(raw)
    except Exception as e:
        logger.warning(f"[extract_slots_llm] failed: {e}")
        return {
            "incident_date": None, "incident_time": None, "incident_place": None,
            "accused": None, "witnesses": None, "injury": None,
            "property_loss": None, "words_or_threats": None,
            "incident_types": [],  # empty so keyword classifier kicks in
            "description": narrative[:400],
        }


# ─── Correction extraction (DHARA-B012) ───────────────────────────────────────
def _make_correction_sys() -> str:
    today_str = date.today().strftime("%d %B %Y")
    return f"""You are a correction-intake assistant for an Indian FIR complaint draft.
The citizen is correcting one or more previously given facts. Read their correction
instruction and return ONLY valid JSON (no markdown, no explanation) containing ONLY
the fields that should change — omit every field that is not explicitly mentioned.
Possible fields:
{{
  "informant_name": null,
  "incident_date": null,
  "incident_time": null,
  "incident_place_text": null,
  "accused": null,
  "injury": null,
  "force_used": null,
  "property_loss": null
}}
For injury/force_used/property_loss use exactly the string "yes" or "no" — never any
other value — and only include the field if the citizen's correction clearly states it.
Today's date is {today_str}. Convert any date mentioned into YYYY-MM-DD format.
Do not invent or guess a value for a field the citizen did not mention.
"""  # noqa: E501


async def extract_correction_llm(correction_text: str, llm_key: str) -> dict:
    """Extract ONLY the explicitly-corrected canonical fields from a free-text
    correction sentence. Never returns the raw sentence itself as a value."""
    try:
        result = await send_once(
            f"fir-correct-{uuid.uuid4()}",
            _make_correction_sys(),
            correction_text[:800],
            spec=FIR_CORRECTION_SPEC,
            caller="fir_extract_correction",
        )
        raw = (result or "").strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[^\n]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw.strip())
        data = json.loads(raw)
        return {k: v for k, v in data.items() if v is not None and str(v).strip() != ""}
    except Exception as e:
        logger.warning(f"[extract_correction_llm] failed: {e}")
        return {}


# ─── Section suggestion with Citation Guard ───────────────────────────────────
async def suggest_sections(
    corpus_db, incident_types: list[str], slots: dict, language: str = "en"
) -> tuple[list[dict], list[dict]]:
    """
    Look up BNS sections from DB based on incident types (Citation Guard).
    Apply consistency check — drop sections that contradict user input.
    Returns (confirmed_sections, dropped_sections) where each dropped entry
    has {section_number, reason_dropped}.
    Headings are translated to `language` when a translation exists.
    """
    from corpus_db import lookup_section as db_lookup

    # Gather candidate section IDs
    candidates: set[str] = set()
    for t in incident_types:
        candidates.update(INCIDENT_SECTIONS.get(t, []))

    # ── Consistency check with reason tracking ────────────────────────────
    dropped_ids: list[dict] = []
    if str(slots.get("injury", "")).lower().startswith("no"):
        for sec_id in _INJURY_REQUIRED & candidates:
            dropped_ids.append({"section_number": sec_id, "reason_dropped": "contradicts injury=none"})
        candidates -= _INJURY_REQUIRED
    if str(slots.get("property_loss", "")).lower().startswith("no"):
        for sec_id in _PROPERTY_REQUIRED & candidates:
            dropped_ids.append({"section_number": sec_id, "reason_dropped": "contradicts property_loss=none"})
        candidates -= _PROPERTY_REQUIRED
    if str(slots.get("force_used", "")).lower().startswith("no"):
        for sec_id in _FORCE_REQUIRED & candidates:
            dropped_ids.append({"section_number": sec_id, "reason_dropped": "contradicts force_used=none"})
        candidates -= _FORCE_REQUIRED

    confirmed: list[dict] = []
    for sec_id in sorted(candidates):
        try:
            doc = await db_lookup(corpus_db, sec_id, act_hint="Bharatiya Nyaya Sanhita")
            if doc and not doc.get("is_dead_law") and not doc.get("dead_warning"):
                raw_heading = doc.get("section_heading", "")
                confirmed.append({
                    "section_number": doc.get("section_number", sec_id),
                    "section_heading": _translate_bns_heading(
                        doc.get("section_number", sec_id), raw_heading, language
                    ),
                    "section_heading_en": raw_heading,   # always keep English for the draft
                    "act_name": doc.get("act_name", "Bharatiya Nyaya Sanhita, 2023"),
                    "judicial_flag": doc.get("judicial_flag"),
                })
        except Exception as exc:
            logger.warning(f"[suggest_sections] lookup {sec_id} failed: {exc}")

    return confirmed, dropped_ids


# ─── Draft generation ─────────────────────────────────────────────────────────
_DRAFT_SYS = """You are a legal draft writer for Indian citizens.
Generate a first-person FIR complaint letter as required for Section 173(1) BNSS filing.
Be factual. Use exactly the information provided. Output ONLY the letter text, no markdown."""


async def generate_draft(
    slots: dict,
    sections: list[dict],
    evidence_files: list[dict],
    language: str,
    language_name: str,
    llm_key: str,
) -> str:
    """Generate bilingual FIR complaint letter using Claude Sonnet."""
    sec_text = "\n".join(
        f"• BNS Section {s['section_number']}: {s['section_heading']}" for s in sections
    ) or "• To be determined by the investigating officer"
    ev_text = "\n".join(
        f"• Exhibit {i+1}: {e.get('filename', 'File')}" +
        (f" — {e.get('caption')}" if e.get('caption') else f" ({e.get('file_type', 'document')})")
        for i, e in enumerate(evidence_files)
    ) or "None attached at this stage"
    loc = (slots.get("incident_gps_address") or slots.get("incident_place_text") or "[Not provided]")
    date_str = datetime.now(timezone.utc).strftime("%d %B %Y")

    # Build witness text (combine witnesses + witnesses_detail)
    witnesses_val = slots.get("witnesses", "None mentioned")
    witnesses_detail = slots.get("witnesses_detail")
    if witnesses_detail and witnesses_detail.lower() not in ["not available right now", "skip", "n/a"]:
        witnesses_val = f"{witnesses_val}; {witnesses_detail}"

    lang_line = (
        f"Then repeat the ENTIRE letter in {language_name}, "
        f"preceded by: --- {language_name.upper()} TRANSLATION ---"
        if language not in ("en", "english") else ""
    )

    # IMPORTANT: Only include incident_place_text if it actually looks like a place.
    # Off-topic answers stored in the field are filtered here as a final safety net.
    place_display = loc
    if loc and not _is_valid_place_response(loc):
        place_display = "[To be provided at the time of statement recording]"

    prompt = f"""Generate a formal FIR complaint letter with the following details:

Complainant name: {slots.get('informant_name') or 'Not provided'}
Complainant address: {slots.get('informant_address') or 'Not provided'}
Contact number: {slots.get('informant_phone') or 'Not provided'}
Date of incident: {slots.get('incident_date') or 'Not specified'}
Time of incident: {slots.get('incident_time') or 'Not specified'}
Place of incident: {place_display}
Incident description: {slots.get('description') or 'Not provided'}
Accused: {slots.get('accused') or 'Not identified'}
Injury: {slots.get('injury') or 'Not reported'}
Property loss / stolen items: {slots.get('property_loss') or 'Not reported'}{' | ' + str(slots.get('theft_items','')) if slots.get('theft_items') else ''}
Force used: {slots.get('force_used') or 'Not reported'}
IMEI (if phone stolen): {slots.get('stolen_phone_imei') or ''}
Transaction IDs: {slots.get('transaction_ids') or ''}
Scammer contact: {slots.get('scammer_contact') or ''}
Additional details: {slots.get('cyber_amount', '')} {slots.get('assault_mlc', '')} {slots.get('harassment_mode', '')}
Witnesses: {witnesses_val}

Applicable BNS sections (suggested):
{sec_text}

Evidence attached:
{ev_text}

Date of report: {date_str}

Write the letter in English, addressed to "The Station House Officer".
Write in first person throughout.
IMPORTANT: NEVER use [bracket placeholders] in the letter. If a detail is missing or not provided, write "Not provided" or omit the sentence entirely. Do not leave any square bracket text in the output.
IMPORTANT: If Injury, Force used, or Property loss above are stated as "no", explicitly state that none occurred (e.g. "No injuries were sustained," "No force or weapon was used"). Never imply force, a weapon, or an injury happened unless the corresponding field above says "yes".
{lang_line}

End the letter (after both languages if bilingual) with EXACTLY this disclaimer block:
---
\u26a0\ufe0f CITIZEN DRAFT \u2014 NOT A REGISTERED FIR
Present this document at the nearest police station for official registration under Section 173(1) BNSS.
Suggested BNS sections are indicative only \u2014 the investigating officer determines final sections.
FREE LEGAL AID: NALSA 15100 | Women Helpline 181 | Cybercrime 1930 | Police 100"""

    try:
        reply = await send_once(
            f"fir-draft-{uuid.uuid4()}",
            _DRAFT_SYS,
            prompt,
            spec=FIR_DRAFT_SPEC,
            caller="fir_generate_draft",
        )
        return (reply or "").strip()
    except Exception as e:
        logger.error(f"[generate_draft] LLM failed: {e}")
        return _fallback_draft(slots, sec_text, ev_text, date_str, place_display)


def _fallback_draft(slots: dict, sec_text: str, ev_text: str, date_str: str, loc: str) -> str:
    name = slots.get('informant_name') or 'Complainant'
    address = slots.get('informant_address') or 'Address not provided'
    phone = slots.get('informant_phone') or 'Not provided'
    inc_date = slots.get('incident_date') or 'Date not provided'
    inc_time = slots.get('incident_time') or 'Time not provided'
    desc = slots.get('description') or 'Details to be provided at the time of statement'
    accused = slots.get('accused') or 'Not identified'
    injury = slots.get('injury') or 'Not reported'
    witnesses = slots.get('witnesses') or 'None mentioned'
    return f"""To,
The Station House Officer,
(Police Station to be filled)

Subject: Complaint regarding {desc[:80]}

Sir/Madam,

I, {name}, wish to register the following complaint.

Address: {address}
Contact: {phone}

On {inc_date} at {inc_time},
at {loc}, the following occurred:

{desc}

Accused: {accused}
Injury: {injury}
Witnesses: {witnesses}

Evidence attached:
{ev_text}

Applicable BNS sections (suggested):
{sec_text}

I request you to register this FIR and take appropriate legal action.

Yours faithfully,
{name}
Date: {date_str}

---
\u26a0\ufe0f CITIZEN DRAFT \u2014 NOT A REGISTERED FIR
Present at the police station for registration under Section 173(1) BNSS.
FREE LEGAL AID: NALSA 15100 | Women Helpline 181 | Cybercrime 1930 | Police 100"""


# ─── Summary builder ──────────────────────────────────────────────────────────
def _build_summary(slots: dict, sections: list[dict], incident_types: list[str]) -> str:
    lines: list[str] = []
    dt = (slots.get("incident_date") or "") + (" " + slots.get("incident_time", "") if slots.get("incident_time") else "")
    if dt.strip():
        lines.append(f"\U0001f4c5 When: {dt.strip()}")
    loc = slots.get("incident_gps_address") or slots.get("incident_place_text")
    if loc and _is_valid_place_response(str(loc)):
        lines.append(f"\U0001f4cd Where: {str(loc)[:120]}")
    elif loc:
        lines.append("\U0001f4cd Where: To be confirmed at the time of statement")
    # DHARA-B010/B011: only show an incident type if a surviving (non-dropped)
    # confirmed section actually supports it — e.g. once BNS 309 is dropped
    # because the citizen said "no force", "Robbery" must not still appear
    # here even though the LLM/keyword classifier first guessed it.
    confirmed_ids = {s.get("section_number") for s in sections}
    shown_types = [
        t for t in incident_types
        if not INCIDENT_SECTIONS.get(t) or (set(INCIDENT_SECTIONS.get(t, [])) & confirmed_ids)
    ]
    if shown_types:
        lines.append(f"\u2696\ufe0f Type: {', '.join(t.replace('_', ' ').title() for t in shown_types)}")
    if slots.get("accused"):
        lines.append(f"\U0001f464 Accused: {str(slots.get('accused', ''))[:100]}")
    if slots.get("injury"):
        lines.append(f"\U0001f3e5 Injury: {slots.get('injury')}")
    if slots.get("informant_name"):
        lines.append(f"\U0001f4dd Complainant: {slots.get('informant_name')}")
    if sections:
        sec_list = ", ".join(f"BNS {s['section_number']}" for s in sections[:4])
        lines.append(f"\U0001f4cb Suggested sections: {sec_list}")
    return "\n".join(lines) or "Summary not available"


# ─── Session creation ─────────────────────────────────────────────────────────
async def create_session(
    db,
    user_id: str,
    language: str,
    session_location_start: Optional[dict] = None,
    incident_type: Optional[str] = None,
) -> dict:
    """Create a fresh, blank FIR session. No carryover from any previous session."""
    sid = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    blank_slots = {
        "incident_date": None, "incident_time": None,
        "incident_place_text": None, "incident_gps": None, "incident_gps_address": None,
        "description": None, "accused": None, "witnesses": None, "witnesses_detail": None,
        "injury": None, "property_loss": None, "words_or_threats": None,
        "theft_items": None, "cyber_amount": None, "assault_mlc": None, "harassment_mode": None,
        "force_used": None, "stolen_phone_imei": None, "sim_blocked": None,
        "incident_place_detail": None, "transaction_ids": None, "scammer_contact": None,
        "date_confirmed": None,
        "informant_name": None, "informant_address": None, "informant_phone": None,
        # New module-specific slots
        "dv_duration": None, "dv_children": None,
        "posh_employer": None, "posh_role": None, "posh_icc": None,
        "consumer_company": None, "consumer_order_id": None,
    }
    # Pre-seed incident type if user selected a module
    pre_seeded_types: list[str] = []
    if incident_type and incident_type not in ("other", "Other", ""):
        # Map frontend module keys to engine incident type strings
        _type_map = {
            "cybercrime": "cyber_fraud",
            "cyber_fraud": "cyber_fraud",
            "domestic_violence": "domestic_violence",
            "theft": "theft",
            "posh": "posh",
            "consumer_fraud": "consumer_fraud",
        }
        mapped = _type_map.get(incident_type.lower(), incident_type.lower())
        pre_seeded_types = [mapped]

    doc = {
        "session_id": sid,
        "user_id": user_id,
        "language": language,
        "status": "active",
        "stage": STAGE_SAFETY_GATE,
        "pending_probes": [],
        "current_probe": None,
        "session_location_start": session_location_start,
        "session_location_end": None,
        "incident_types": pre_seeded_types,
        "narrative_turns": [],
        "slots": blank_slots,
        "evidence_files": [],
        "suggested_sections": [],
        "dropped_sections": [],
        "safety_flags": [],
        "parked_questions": [],        # v3.1: off-topic user responses parked here
        "date_conflict_msg": None,     # v3.1: set when contradiction detected
        "relative_date_display": None, # v3.3: absolute date computed from relative
        "phone_retry_done": False,     # v3.1: one phone retry flag
        "probe_history": [],           # v3.3: [{probe, slot, value}] for back navigation
        "draft": None,
        "created_at": now,
        "updated_at": now,
    }
    await db.fir_sessions.insert_one(doc)

    # Build welcome message — module-specific if pre-selected, generic otherwise
    safe_q = _tm("safe_prompt", language)
    if pre_seeded_types:
        module_key = incident_type.lower() if incident_type else ""
        lang_msg = _MODULE_WELCOME.get(module_key, {}).get(language) or \
                   _MODULE_WELCOME.get(module_key, {}).get("en", "")
        if lang_msg:
            bot_message = f"{lang_msg}\n\n🔒 {safe_q}"
        else:
            bot_message = (
                "Hello! I'm DHARA, your legal assistant.\n"
                "I'll help you prepare a formal FIR draft step by step.\n\n"
                f"{safe_q}"
            )
    else:
        bot_message = (
            "Hello! I'm DHARA, your legal assistant.\n"
            "I'll help you prepare a formal FIR draft step by step.\n\n"
            f"{safe_q}"
        )

    return {
        "session_id": sid,
        "stage": STAGE_SAFETY_GATE,
        "bot_message": bot_message,
        "input_type": INPUT_QUICK_REPLY,
        "quick_replies": [_tm("safe_yes", language), _tm("safe_no", language)],
        "safety_flags": [],
        "completed": False,
    }


# ─── Main turn processor ──────────────────────────────────────────────────────
async def process_turn(
    db,
    corpus_db,
    session_id: str,
    user_message: Optional[str],
    gps: Optional[dict],
    action: Optional[str],
    llm_key: str,
) -> dict:
    """Process one conversation turn. Returns the next bot response dict."""
    now = datetime.now(timezone.utc).isoformat()
    session = await db.fir_sessions.find_one({"session_id": session_id})
    if not session:
        return {"error": "Session not found"}

    language: str = session.get("language", "en")
    lang_map = {
        "en": "English", "hi": "Hindi", "ta": "Tamil",
        "mr": "Marathi", "te": "Telugu", "kn": "Kannada",
    }
    language_name = lang_map.get(language, "English")

    stage: str = session.get("stage", STAGE_SAFETY_GATE)
    slots: dict = dict(session.get("slots", {}))
    pending_probes: list[str] = list(session.get("pending_probes", []))
    current_probe: Optional[str] = session.get("current_probe")
    incident_types: list[str] = list(session.get("incident_types", []))

    # Log user turn to conversation history
    if user_message or gps or action:
        msg_log = user_message or (f"[GPS:{gps}]" if gps else f"[{action}]")
        await db.fir_sessions.update_one(
            {"session_id": session_id},
            {"$push": {"narrative_turns": {"role": "user", "message": msg_log, "timestamp": now}},
             "$set": {"updated_at": now}},
        )

    resp = await _dispatch(
        db, corpus_db, session, stage, slots, pending_probes, current_probe,
        incident_types, user_message, gps, action, llm_key, language, language_name, now,
    )

    # Log bot response to conversation history
    if resp.get("bot_message"):
        await db.fir_sessions.update_one(
            {"session_id": session_id},
            {"$push": {"narrative_turns": {
                "role": "bot", "message": resp["bot_message"], "timestamp": now,
            }}},
        )
    return resp


async def _dispatch(
    db, corpus_db, session, stage, slots, pending_probes, current_probe,
    incident_types, user_message, gps, action, llm_key, language, language_name, now,
) -> dict:
    sid: str = session["session_id"]

    # ── Safety Gate ───────────────────────────────────────────────────────────
    if stage == STAGE_SAFETY_GATE:
        sf = _detect_safety_flags(user_message or "")
        # Use boolean flag: action=="not_safe" is the authoritative signal
        # (frontend sends this when "No, I need help" button is tapped)
        not_safe = (
            action == "not_safe"
            or (user_message or "").lower().strip() in [
                "no", "not safe", "in danger", "help me",
                "no i need help", "no, i need help",
            ]
        )
        if not_safe:
            # Pause session so it appears in "Continue your reports"
            await db.fir_sessions.update_one(
                {"session_id": sid},
                {"$set": {"status": "paused", "safety_flags": sf, "updated_at": now}},
            )
            return {
                "session_id": sid, "stage": STAGE_SAFETY_GATE,
                "bot_message": _tm("not_safe_msg", language),
                "input_type": INPUT_QUICK_REPLY,
                "quick_replies": [_tm("safe_continue", language), _tm("exit_now", language)],
                "show_emergency": True,
                "emergency_numbers": [
                    {"label": "Police / Emergency", "number": "112"},
                    {"label": "Women Helpline", "number": "181"},
                    {"label": "Ambulance", "number": "108"},
                    {"label": "NALSA Legal Aid", "number": "15100"},
                ],
                "safety_flags": sf, "completed": False,
            }

        lines = [_tm("trust_msg", language)]
        if "POCSO" in sf:
            lines.append(_tm("pocso_alert", language))
        elif sf:
            lines.append(_tm("dv_alert", language))
        lines.append(_tm("what_happened", language))
        await db.fir_sessions.update_one(
            {"session_id": sid},
            {"$set": {"stage": STAGE_FREE_NARRATIVE, "safety_flags": sf, "updated_at": now}},
        )
        return {
            "session_id": sid, "stage": STAGE_FREE_NARRATIVE,
            "bot_message": "\n".join(lines),
            "input_type": INPUT_VOICE_TEXT, "quick_replies": [],
            "safety_flags": sf, "completed": False,
        }

    # ── Free Narrative ────────────────────────────────────────────────────────
    elif stage == STAGE_FREE_NARRATIVE:
        narrative = (user_message or "").strip()

        # ── v3.3: Handle safety gate "I'm safe now" continuation ───────────
        if action == "continue_safe":
            await db.fir_sessions.update_one(
                {"session_id": sid},
                {"$set": {"stage": STAGE_FREE_NARRATIVE, "status": "active", "updated_at": now}},
            )
            return {
                "session_id": sid, "stage": STAGE_FREE_NARRATIVE,
                "bot_message": _tm("safe_now_continue", language),
                "input_type": INPUT_VOICE_TEXT, "quick_replies": [], "completed": False,
            }

        if len(narrative) < 20:
            return {
                "session_id": sid, "stage": STAGE_FREE_NARRATIVE,
                "bot_message": _tm("more_detail", language),
                "input_type": INPUT_VOICE_TEXT, "quick_replies": [], "completed": False,
            }

        # ── v3.3: Immediate danger detection (EMERGENCY) ────────────────────
        if _detect_immediate_danger(narrative):
            await db.fir_sessions.update_one(
                {"session_id": sid},
                {"$set": {"status": "paused", "updated_at": now}},
            )
            return {
                "session_id": sid, "stage": STAGE_FREE_NARRATIVE,
                "bot_message": _tm("emergency_danger", language),
                "input_type": INPUT_QUICK_REPLY,
                "quick_replies": [_tm("safe_continue_short", language)],
                "show_emergency": True,
                "action": "EMERGENCY",
                "emergency_numbers": [
                    {"label": "Emergency", "number": "112"},
                    {"label": "Women Helpline", "number": "181"},
                    {"label": "Ambulance", "number": "108"},
                ],
                "completed": False,
            }

        extracted = await extract_slots_llm(narrative, llm_key)

        # ── FIX 8: Keyword fallback + merge for incident_types ──────────────
        raw_types = extracted.get("incident_types") or []
        kw_types = _keyword_classify(narrative)
        if not raw_types or raw_types == ["other"]:
            # LLM gave nothing useful — use keyword results
            raw_types = kw_types
        else:
            # LLM gave results — merge with keyword to catch what LLM missed
            # DHARA-P0-1: only strong (non-generic) keywords may add a type
            kw_strong = _keyword_classify(narrative, strong_only=True)
            merged = list(set(raw_types) | (set(kw_strong) - {"other"}))
            raw_types = merged if merged else raw_types
        new_types = raw_types if raw_types else ["other"]

        # Merge extracted slots (never overwrite existing)
        for key in ["incident_date", "incident_time", "accused", "witnesses",
                    "injury", "property_loss", "words_or_threats", "description"]:
            if extracted.get(key) is not None and not slots.get(key):
                slots[key] = extracted[key]
        if extracted.get("incident_place") and not slots.get("incident_place_text"):
            slots["incident_place_text"] = extracted["incident_place"]

        # ── FIX 1: Date contradiction detection ─────────────────────────────
        date_conflict_msg = _has_date_conflict(narrative, extracted.get("incident_date"))
        if date_conflict_msg:
            slots["incident_date"] = None

        # ── v3.3: Relative date tracking — build confirm message ─────────────
        relative_date_display: Optional[str] = None
        t_low = narrative.lower()
        if extracted.get("incident_date") and not date_conflict_msg:
            if any(kw in t_low for kw in _RELATIVE_DATE_KW.get("today", [])):
                relative_date_display = f"today, {extracted['incident_date']}"
            elif any(kw in t_low for kw in _RELATIVE_DATE_KW.get("yesterday", [])):
                relative_date_display = f"yesterday, {extracted['incident_date']}"

        sf_all = list(set(list(session.get("safety_flags", [])) + _detect_safety_flags(narrative)))
        probe_queue = _build_probe_queue(
            slots, new_types, date_conflict_msg, relative_date_display
        )
        first_probe = probe_queue[0] if probe_queue else None
        remaining = probe_queue[1:] if probe_queue else []

        # ── v3.3: Cybercrime alert for first probe ───────────────────────────
        show_cyber_alert = "cyber_fraud" in new_types

        await db.fir_sessions.update_one(
            {"session_id": sid},
            {"$set": {
                "stage": STAGE_PROBE if first_probe else STAGE_SECTION_SUGGEST,
                "incident_types": new_types, "slots": slots,
                "pending_probes": remaining, "current_probe": first_probe,
                "safety_flags": sf_all,
                "date_conflict_msg": date_conflict_msg,
                "relative_date_display": relative_date_display,
                "updated_at": now,
            }},
        )
        if first_probe:
            types_str = " and ".join(t.replace("_", " ") for t in new_types[:2])
            pdef = PROBE_Q[first_probe]

            # Dynamic message for probe_date_confirm
            if first_probe == "probe_date_confirm" and relative_date_display:
                bot_msg_suffix = (
                    f"I calculated the incident happened on **{relative_date_display}**.\n"
                    "Is that correct?"
                )
            elif first_probe == "probe_date" and date_conflict_msg:
                bot_msg_suffix = date_conflict_msg
            else:
                bot_msg_suffix = _tp(first_probe, language)

            # Cybercrime helpline header
            cyber_header = _tm("cybercrime_alert", language) if show_cyber_alert else ""

            intro = _tm("thanks_sharing", language, types_str=types_str, suffix=f"{cyber_header}{bot_msg_suffix}")
            return {
                "session_id": sid, "stage": STAGE_PROBE, "probe_key": first_probe,
                "bot_message": intro,
                "input_type": pdef["input_type"],
                "quick_replies": pdef.get("quick_replies", []),
                "skip_label": pdef.get("skip_label"),
                "safety_flags": sf_all,
                "show_cybercrime_alert": show_cyber_alert,
                "completed": False,
            }
        return await _enter_section_suggest(db, corpus_db, sid, slots, new_types, now, language)

    # ── Probe ─────────────────────────────────────────────────────────────────
    elif stage == STAGE_PROBE:
        cp = current_probe
        if not cp:
            return await _enter_section_suggest(db, corpus_db, sid, slots, incident_types, now, language)
        pdef = PROBE_Q.get(cp, {})
        slot_key = pdef.get("slot")

        # ── v3.3: Resume action — re-ask current probe without advancing ──────
        if action == "resume":
            session_fresh = await db.fir_sessions.find_one({"session_id": sid}, {"relative_date_display": 1})
            rdd = (session_fresh or {}).get("relative_date_display")
            if cp == "probe_date_confirm" and rdd:
                bot_msg = _tm("welcome_back", language, suffix=(
                    f"I calculated the incident happened on **{rdd}**.\nIs that correct?"
                ))
            else:
                bot_msg = _tm("welcome_back", language, suffix=_tp(cp, language))
            # v3.4: Probe progress for "Q X/Y" UI counter
            done_count = len(session.get("probe_history", []))
            total_probes = done_count + 1 + len(pending_probes)
            return {
                "session_id": sid, "stage": STAGE_PROBE, "probe_key": cp,
                "bot_message": bot_msg,
                "input_type": pdef.get("input_type", INPUT_TEXT),
                "quick_replies": pdef.get("quick_replies", []),
                "skip_label": pdef.get("skip_label"),
                "completed": False,
                "probe_progress_done": done_count,
                "probe_progress_total": total_probes,
            }

        if cp == "probe_place_gps":
            if gps and action != "skip":
                address = await reverse_geocode(gps["lat"], gps["lng"])
                addr = address or f"{gps['lat']:.4f}, {gps['lng']:.4f}"
                slots["incident_gps"] = gps
                slots["incident_gps_address"] = addr
                await db.fir_sessions.update_one(
                    {"session_id": sid},
                    {"$set": {"stage": STAGE_GPS_CONFIRM, "slots": slots, "updated_at": now}},
                )
                return {
                    "session_id": sid, "stage": STAGE_GPS_CONFIRM,
                    "bot_message": _tm("gps_found", language, addr=addr),
                    "input_type": INPUT_QUICK_REPLY,
                    "quick_replies": [_tm("gps_yes", language), _tm("gps_no", language)],
                    "confirmed_address": addr, "completed": False,
                }
            else:
                # ── v3.3: GPS declined/skipped — record source ───────────────
                slots["incident_gps"] = {"source": "skipped"}
                await db.fir_sessions.update_one(
                    {"session_id": sid},
                    {"$set": {"slots": slots, "updated_at": now}},
                )
            return await _advance_probe(db, corpus_db, sid, slots, pending_probes, incident_types, now, completed_probe=cp, language=language)

        elif cp == "probe_evidence":
            if action in ("skip", "upload_done"):
                return await _advance_probe(db, corpus_db, sid, slots, pending_probes, incident_types, now, language=language)
            return {
                "session_id": sid, "stage": STAGE_PROBE, "probe_key": cp,
                "bot_message": _tp(cp, language), "input_type": INPUT_EVIDENCE,
                "skip_label": pdef.get("skip_label"), "quick_replies": [], "completed": False,
            }

        elif slot_key:
            if action == "skip":
                pass  # advance to next probe

            elif user_message:
                # ── FIX 2 & 3: Place validation + parked_questions ───────────
                if cp == "probe_place_text":
                    if _is_valid_place_response(user_message):
                        slots["incident_place_text"] = user_message
                        await db.fir_sessions.update_one(
                            {"session_id": sid}, {"$set": {"slots": slots, "updated_at": now}}
                        )
                    else:
                        # Park the off-topic response and RE-ASK the same probe
                        await db.fir_sessions.update_one(
                            {"session_id": sid},
                            {
                                "$push": {"parked_questions": {
                                    "probe": cp,
                                    "user_response": user_message,
                                    "reason": "off_topic",
                                    "timestamp": now,
                                }},
                                "$set": {"updated_at": now},
                            },
                        )
                        # Return the same probe question again with a note
                        return {
                            "session_id": sid, "stage": STAGE_PROBE, "probe_key": cp,
                            "bot_message": _tm("place_off_topic", language, suffix=_tp(cp, language)),
                            "input_type": pdef["input_type"],
                            "quick_replies": pdef.get("quick_replies", []),
                            "skip_label": pdef.get("skip_label"),
                            "completed": False,
                        }

                # ── DHARA-B009: Date/time validation before it reaches a draft ──
                elif cp == "probe_date":
                    date_err = _date_validation_error(user_message)
                    if date_err:
                        return {
                            "session_id": sid, "stage": STAGE_PROBE, "probe_key": cp,
                            "bot_message": date_err,
                            "input_type": pdef["input_type"],
                            "quick_replies": pdef.get("quick_replies", []),
                            "skip_label": pdef.get("skip_label"), "completed": False,
                        }
                    slots["incident_date"] = user_message
                    await db.fir_sessions.update_one(
                        {"session_id": sid}, {"$set": {"slots": slots, "updated_at": now}}
                    )

                elif cp == "probe_time":
                    if not _validate_time_input(user_message):
                        return {
                            "session_id": sid, "stage": STAGE_PROBE, "probe_key": cp,
                            "bot_message": "That time doesn't look valid — please use HH:MM (e.g. 14:30 or 2:30 PM).",
                            "input_type": pdef["input_type"],
                            "quick_replies": pdef.get("quick_replies", []),
                            "skip_label": pdef.get("skip_label"), "completed": False,
                        }
                    slots["incident_time"] = user_message
                    await db.fir_sessions.update_one(
                        {"session_id": sid}, {"$set": {"slots": slots, "updated_at": now}}
                    )

                # ── FIX 4: Accused slot guard ────────────────────────────────
                elif cp == "probe_accused":
                    if _is_witness_answer_in_accused_probe(user_message):
                        # User is answering about witnesses, not accused
                        # Route to witnesses slot if not already set
                        if not slots.get("witnesses"):
                            slots["witnesses"] = user_message
                        slots["accused"] = "Not identified by complainant"
                        await db.fir_sessions.update_one(
                            {"session_id": sid}, {"$set": {"slots": slots, "updated_at": now}}
                        )
                    else:
                        slots["accused"] = user_message
                        await db.fir_sessions.update_one(
                            {"session_id": sid}, {"$set": {"slots": slots, "updated_at": now}}
                        )

                # ── FIX 5: Witness vague answer → follow-up probe ─────────────
                elif cp == "probe_witnesses":
                    slots["witnesses"] = user_message
                    await db.fir_sessions.update_one(
                        {"session_id": sid}, {"$set": {"slots": slots, "updated_at": now}}
                    )
                    if _is_vague_witness_response(user_message):
                        # Insert detail follow-up NEXT in the queue
                        new_pending = ["probe_witnesses_detail"] + pending_probes
                        await db.fir_sessions.update_one(
                            {"session_id": sid},
                            {"$set": {"pending_probes": new_pending, "updated_at": now}}
                        )
                        return await _advance_probe(
                            db, corpus_db, sid, slots, new_pending, incident_types, now, language=language
                        )

                elif cp == "probe_witnesses_detail":
                    # Append detail to existing witnesses string
                    existing = slots.get("witnesses", "")
                    slots["witnesses_detail"] = user_message
                    await db.fir_sessions.update_one(
                        {"session_id": sid}, {"$set": {"slots": slots, "updated_at": now}}
                    )

                # ── FIX 6: Phone validation + one retry ──────────────────────
                elif cp == "probe_informant_phone":
                    if not _is_valid_phone(user_message):
                        phone_retry_done = session.get("phone_retry_done", False)
                        if not phone_retry_done:
                            # One retry
                            await db.fir_sessions.update_one(
                                {"session_id": sid},
                                {"$set": {
                                    "phone_retry_done": True,
                                    "pending_probes": ["probe_informant_phone_retry"] + pending_probes,
                                    "updated_at": now,
                                }},
                            )
                            return {
                                "session_id": sid, "stage": STAGE_PROBE,
                                "probe_key": "probe_informant_phone_retry",
                                "bot_message": _tp("probe_informant_phone_retry", language),
                                "input_type": INPUT_TEXT,
                                "skip_label": PROBE_Q["probe_informant_phone_retry"]["skip_label"],
                                "quick_replies": [], "completed": False,
                            }
                        else:
                            # Second attempt — accept whatever they give (maybe they have a reason)
                            slots["informant_phone"] = user_message
                            await db.fir_sessions.update_one(
                                {"session_id": sid}, {"$set": {"slots": slots, "updated_at": now}}
                            )
                    else:
                        slots["informant_phone"] = user_message
                        await db.fir_sessions.update_one(
                            {"session_id": sid}, {"$set": {"slots": slots, "updated_at": now}}
                        )

                elif cp == "probe_informant_phone_retry":
                    # Accept whatever is given at this point
                    slots["informant_phone"] = user_message
                    await db.fir_sessions.update_one(
                        {"session_id": sid}, {"$set": {"slots": slots, "updated_at": now}}
                    )

                elif cp == "probe_injury":
                    # DHARA-B011: the old check `"injur" in ml` matched the
                    # substring "injur" inside "No — no injuries" itself,
                    # silently flipping an explicit negative answer to "yes".
                    # Quick-reply options always start with "Yes"/"No" — use
                    # that deterministically instead of substring matching.
                    ml = user_message.lower().strip()
                    slots["injury"] = "yes" if _is_affirmative(user_message) else "no"  # DHARA-P0-2
                    if slots["injury"] == "no":
                        # DHARA-B010: don't ask for a Medico-Legal Certificate
                        # once the citizen has already said no one was injured.
                        pending_probes = [p for p in pending_probes if p != "probe_assault_mlc"]
                    await db.fir_sessions.update_one(
                        {"session_id": sid}, {"$set": {"slots": slots, "pending_probes": pending_probes, "updated_at": now}}
                    )
                elif cp == "probe_harassment_online":
                    slots["harassment_mode"] = user_message
                    await db.fir_sessions.update_one(
                        {"session_id": sid}, {"$set": {"slots": slots, "updated_at": now}}
                    )
                # ── v3.3: New probe handlers ─────────────────────────────────
                elif cp == "probe_force_used":
                    # DHARA-B010/B011: the old check `"force" in m_lower`
                    # matched the substring "force" inside "No, no force"
                    # itself, flipping an explicit negative into force=yes —
                    # which then wrongly kept BNS 309 Robbery suggested.
                    m_lower = user_message.lower().strip()
                    force = _is_affirmative(user_message)  # DHARA-P0-2
                    slots["force_used"] = "yes" if force else "no"
                    await db.fir_sessions.update_one(
                        {"session_id": sid}, {"$set": {"slots": slots, "updated_at": now}}
                    )
                    # Robbery upgrade: if force confirmed AND theft, add robbery
                    if force and "theft" in incident_types and "robbery" not in incident_types:
                        new_types = list(incident_types) + ["robbery"]
                        # Re-run section retrieval
                        new_sections, _ = await suggest_sections(corpus_db, new_types, slots, language)
                        # Insert injury + mlc probes if not already in queue
                        extra = []
                        if "probe_injury" not in pending_probes and not slots.get("injury"):
                            extra.append("probe_injury")
                        if "probe_assault_mlc" not in pending_probes:
                            extra.append("probe_assault_mlc")
                        new_pending = extra + pending_probes
                        await db.fir_sessions.update_one(
                            {"session_id": sid},
                            {"$set": {
                                "incident_types": new_types,
                                "suggested_sections": new_sections,
                                "pending_probes": new_pending,
                                "updated_at": now,
                            }}
                        )
                        pending_probes = new_pending
                        incident_types = new_types

                elif cp == "probe_stolen_phone_imei":
                    slots["stolen_phone_imei"] = user_message
                    await db.fir_sessions.update_one(
                        {"session_id": sid}, {"$set": {"slots": slots, "updated_at": now}}
                    )
                elif cp == "probe_sim_blocked":
                    slots["sim_blocked"] = user_message
                    await db.fir_sessions.update_one(
                        {"session_id": sid}, {"$set": {"slots": slots, "updated_at": now}}
                    )
                elif cp == "probe_incident_place_detail":
                    slots["incident_place_detail"] = user_message
                    await db.fir_sessions.update_one(
                        {"session_id": sid}, {"$set": {"slots": slots, "updated_at": now}}
                    )
                elif cp == "probe_transaction_ids":
                    slots["transaction_ids"] = user_message
                    await db.fir_sessions.update_one(
                        {"session_id": sid}, {"$set": {"slots": slots, "updated_at": now}}
                    )
                elif cp == "probe_scammer_contact":
                    slots["scammer_contact"] = user_message
                    await db.fir_sessions.update_one(
                        {"session_id": sid}, {"$set": {"slots": slots, "updated_at": now}}
                    )
                else:
                    slots[slot_key] = user_message
                    await db.fir_sessions.update_one(
                        {"session_id": sid}, {"$set": {"slots": slots, "updated_at": now}}
                    )

        # ── v3.3: Handle probe_date_confirm (quick-reply probe, not text) ───
        if cp == "probe_date_confirm":
            m_lower = (user_message or "").lower()
            if "no" in m_lower or "correct" in m_lower and "yes" not in m_lower:
                # User wants to correct — clear date and prepend probe_date
                slots["incident_date"] = None
                slots["date_confirmed"] = None
                new_pending = ["probe_date"] + pending_probes
                await db.fir_sessions.update_one(
                    {"session_id": sid},
                    {"$set": {"slots": slots, "pending_probes": new_pending, "updated_at": now}}
                )
                return await _advance_probe(db, corpus_db, sid, slots, new_pending, incident_types, now, completed_probe=cp, language=language)
            else:
                # User confirmed — mark as confirmed
                slots["date_confirmed"] = True
                await db.fir_sessions.update_one(
                    {"session_id": sid}, {"$set": {"slots": slots, "updated_at": now}}
                )

        return await _advance_probe(db, corpus_db, sid, slots, pending_probes, incident_types, now, completed_probe=cp, language=language)

    # ── GPS Confirm ───────────────────────────────────────────────────────────
    elif stage == STAGE_GPS_CONFIRM:
        msg_lower = (user_message or "").lower()
        yes = "yes" in msg_lower or action == "confirm"
        if not yes:
            # User rejected GPS address — clear it, keep text description
            slots["incident_gps"] = None
            slots["incident_gps_address"] = None
            await db.fir_sessions.update_one(
                {"session_id": sid},
                {"$set": {"slots": slots, "updated_at": now}},
            )
        # pending_probes in DB still holds remaining probes (those after probe_place_gps)
        session_fresh = await db.fir_sessions.find_one({"session_id": sid})
        pp = list(session_fresh.get("pending_probes", []))
        slots_fresh = dict(session_fresh.get("slots", {}))
        await db.fir_sessions.update_one(
            {"session_id": sid},
            {"$set": {"stage": STAGE_PROBE, "updated_at": now}},
        )
        return await _advance_probe(db, corpus_db, sid, slots_fresh, pp, incident_types, now, language=language)

    # ── Section Suggest ───────────────────────────────────────────────────────
    elif stage == STAGE_SECTION_SUGGEST:
        sections = session.get("suggested_sections", [])
        summary = _build_summary(slots, sections, incident_types)
        await db.fir_sessions.update_one(
            {"session_id": sid}, {"$set": {"stage": STAGE_READ_BACK, "updated_at": now}},
        )
        return {
            "session_id": sid, "stage": STAGE_READ_BACK,
            "bot_message": _tm("summary_correct", language, summary=summary),
            "input_type": INPUT_CONFIRM,
            "quick_replies": [_tm("yes_generate", language), _tm("edit_something", language)],
            "slots_preview": slots, "suggested_sections": sections, "completed": False,
        }

    # ── Read Back ─────────────────────────────────────────────────────────────
    elif stage == STAGE_READ_BACK:
        msg_lower = (user_message or action or "").lower().strip()
        # DHARA-P0-2: also match the translated "Edit something" button and
        # non-English "no" — tapping "कुछ बदलें" used to skip straight to the draft.
        _edit_labels = {v.lower() for v in _STATIC_TRANS.get("edit_something", {}).values()}
        if ("edit" in msg_lower or msg_lower.startswith("no") or _is_negative(msg_lower)
                or any(lbl and lbl in msg_lower for lbl in _edit_labels)):
            # DHARA-B012: move to a dedicated correction stage instead of
            # looping back into STAGE_READ_BACK. The old code re-used this
            # same branch to interpret the *next* message too — a correction
            # sentence that happened to contain "no" (e.g. "no force was
            # used") re-triggered this same prompt forever, and anything
            # else silently skipped straight to draft generation with the
            # correction never applied.
            await db.fir_sessions.update_one(
                {"session_id": sid}, {"$set": {"stage": STAGE_CORRECTION, "updated_at": now}},
            )
            return {
                "session_id": sid, "stage": STAGE_CORRECTION,
                "bot_message": _tm("what_to_change", language),
                "input_type": INPUT_TEXT, "quick_replies": [], "completed": False,
            }
        # Generate draft
        sections = session.get("suggested_sections", [])
        evidence_files = list(session.get("evidence_files", []))
        await db.fir_sessions.update_one(
            {"session_id": sid}, {"$set": {"stage": STAGE_DRAFT, "updated_at": now}},
        )
        draft_text = await generate_draft(
            slots, sections, evidence_files, language, language_name, llm_key
        )
        await db.fir_sessions.update_one(
            {"session_id": sid},
            {"$set": {"draft": draft_text, "stage": STAGE_COMPLETED, "status": "completed", "updated_at": now}},
        )
        return {
            "session_id": sid, "stage": STAGE_COMPLETED,
            "bot_message": _tm("draft_ready", language),
            "input_type": INPUT_DONE, "quick_replies": [],
            "draft": draft_text, "completed": True,
        }

    # ── Correction (DHARA-B012) ────────────────────────────────────────────────
    # Parses a free-text correction into canonical slot fields via a small,
    # targeted LLM extraction — never stores the raw correction sentence as
    # a fact itself — then re-runs the Citation Guard (suggest_sections) so
    # section suggestions stay consistent with the corrected facts, and
    # returns to STAGE_READ_BACK so the citizen reviews the updated summary
    # before a draft is ever generated.
    elif stage == STAGE_CORRECTION:
        correction_text = (user_message or "").strip()
        if len(correction_text) < 2:
            return {
                "session_id": sid, "stage": STAGE_CORRECTION,
                "bot_message": "Please type what you'd like to change (e.g. \"change my name to Beta\" or \"the date was 12/03/2026\").",
                "input_type": INPUT_TEXT, "quick_replies": [], "completed": False,
            }
        updates = await extract_correction_llm(correction_text, llm_key)
        # Validate any corrected date/time deterministically before accepting it
        date_correction_err = None
        if updates.get("incident_date"):
            date_correction_err = _date_validation_error(updates["incident_date"])
            if date_correction_err:
                updates.pop("incident_date", None)
        if updates.get("incident_time") and not _validate_time_input(updates["incident_time"]):
            updates.pop("incident_time", None)
        changed_fields = list(updates.keys())
        for k, v in updates.items():
            if k == "incident_place_text":
                slots["incident_place_text"] = v
                slots["incident_gps_address"] = None  # manual correction overrides any GPS-resolved address
            else:
                slots[k] = v
        # If force_used was corrected to "no", drop robbery from the displayed
        # type immediately (suggest_sections below also drops BNS 309).
        confirmed, dropped = await suggest_sections(corpus_db, incident_types, slots, language)
        await db.fir_sessions.update_one(
            {"session_id": sid},
            {"$set": {
                "slots": slots, "suggested_sections": confirmed, "dropped_sections": dropped,
                "stage": STAGE_READ_BACK, "updated_at": now,
            }},
        )
        summary = _build_summary(slots, confirmed, incident_types)
        if changed_fields:
            changed_note = "Updated: " + ", ".join(f.replace("_", " ") for f in changed_fields)
        else:
            changed_note = "I couldn't find a specific change in that — please tell me exactly what to update (e.g. \"change name to Beta\")."
        if date_correction_err:
            changed_note += f"\n\n{date_correction_err}"
        return {
            "session_id": sid, "stage": STAGE_READ_BACK,
            "bot_message": f"{changed_note}\n\n{_tm('summary_correct', language, summary=summary)}",
            "input_type": INPUT_CONFIRM,
            "quick_replies": [_tm("yes_generate", language), _tm("edit_something", language)],
            "slots_preview": slots, "suggested_sections": confirmed, "completed": False,
        }

    elif stage == STAGE_COMPLETED:
        return {
            "session_id": sid, "stage": STAGE_COMPLETED,
            "bot_message": _tm("draft_already", language),
            "input_type": INPUT_DONE, "draft": session.get("draft", ""),
            "quick_replies": [], "completed": True,
        }

    return {"error": f"Unknown stage: {stage}", "session_id": sid}


async def _advance_probe(
    db, corpus_db, sid: str, slots: dict,
    pending_probes: list[str], incident_types: list[str], now: str,
    completed_probe: Optional[str] = None,
    language: str = "en",
) -> dict:
    """Advance to next probe in queue, or enter section_suggest if none left."""
    # ── v3.3: Track answered probe for back navigation ──────────────────────
    if completed_probe and completed_probe in PROBE_Q:
        pdef_done = PROBE_Q[completed_probe]
        slot_key = pdef_done.get("slot")
        # Store None as "restore value" — going back should clear the slot
        # so the user can re-answer with a fresh prompt
        await db.fir_sessions.update_one(
            {"session_id": sid},
            {"$push": {"probe_history": {"probe": completed_probe, "slot": slot_key, "value": None}}},
        )
    if pending_probes:
        next_probe = pending_probes[0]
        remaining = pending_probes[1:]
        await db.fir_sessions.update_one(
            {"session_id": sid},
            {"$set": {"pending_probes": remaining, "current_probe": next_probe, "updated_at": now}},
        )
        pdef = PROBE_Q[next_probe]
        # Dynamic message for probe_date_confirm; also fetch probe_history for progress counter
        session_fresh = await db.fir_sessions.find_one({"session_id": sid}, {"relative_date_display": 1, "probe_history": 1})
        rdd = session_fresh.get("relative_date_display") if session_fresh else None
        if next_probe == "probe_date_confirm" and rdd:
            bot_msg = f"I calculated the incident happened on **{rdd}**.\nIs that correct?"
        else:
            bot_msg = _tp(next_probe, language)
        # v3.4: Compute probe progress for "Q X/Y" frontend counter
        done_count = len((session_fresh or {}).get("probe_history", []))
        total_probes = done_count + 1 + len(remaining)
        return {
            "session_id": sid, "stage": STAGE_PROBE, "probe_key": next_probe,
            "bot_message": bot_msg,
            "input_type": pdef["input_type"],
            "quick_replies": pdef.get("quick_replies", []),
            "skip_label": pdef.get("skip_label"), "completed": False,
            "probe_progress_done": done_count,
            "probe_progress_total": total_probes,
        }
    return await _enter_section_suggest(db, corpus_db, sid, slots, incident_types, now, language)


async def _enter_section_suggest(
    db, corpus_db, sid: str, slots: dict, incident_types: list[str], now: str, language: str = "en",
) -> dict:
    """Run Citation Guard and enter section_suggest stage."""
    confirmed, dropped = await suggest_sections(corpus_db, incident_types, slots, language)
    await db.fir_sessions.update_one(
        {"session_id": sid},
        {"$set": {
            "stage": STAGE_SECTION_SUGGEST,
            "suggested_sections": confirmed,
            "dropped_sections": dropped,
            "updated_at": now,
        }},
    )
    if confirmed:
        sec_lines = "\n".join(
            f"• BNS {s['section_number']} \u2014 {s['section_heading']}" for s in confirmed[:6]
        )
        msg = _tm("section_found", language, sec_lines=sec_lines)
    else:
        msg = _tm("section_none", language)
    return {
        "session_id": sid, "stage": STAGE_SECTION_SUGGEST,
        "bot_message": msg, "input_type": INPUT_CONFIRM,
        "quick_replies": [_tm("yes_proceed", language), _tm("go_back", language)],
        "suggested_sections": confirmed,
        "dropped_sections": dropped,  # Issue 9: expose dropped sections for drawer
        "completed": False,
    }
