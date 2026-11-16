#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Language Registry module.
Central registry mapping Indian languages to scripts, BCP-47 codes,
FLORES-200/NLLB tokens, IndicTrans2 tokens, and neural TTS voice identifiers.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class VoiceInfo:
    """Metadata describing an available voice model."""
    voice_id: str
    name: str
    gender: str
    engine: str
    sample_rate: int
    is_default: bool = False
    description: str = ""


@dataclass(frozen=True)
class LanguageSpec:
    """Comprehensive metadata for a supported language."""
    code: str                  # ISO-639-1 / standard short code, e.g. 'hi'
    iso_639_3: str             # 3-letter code, e.g. 'hin'
    english_name: str          # 'Hindi'
    native_name: str           # 'हिंदी'
    script_name: str           # 'Devanagari'
    nllb_code: str             # NLLB / FLORES-200 tag, e.g. 'hin_Deva'
    indictrans_code: str       # IndicTrans2 tag, e.g. 'hin_Deva'
    mms_tts_code: str          # Meta MMS-TTS tag, e.g. 'hin'
    indicf5_code: str          # IndicF5 language tag, e.g. 'hi'
    voices: List[VoiceInfo] = field(default_factory=list)


# Authoritative language specifications
LANGUAGES: Dict[str, LanguageSpec] = {
    "hi": LanguageSpec(
        code="hi",
        iso_639_3="hin",
        english_name="Hindi",
        native_name="हिंदी",
        script_name="Devanagari",
        nllb_code="hin_Deva",
        indictrans_code="hin_Deva",
        mms_tts_code="hin",
        indicf5_code="hi",
        voices=[
            VoiceInfo(
                voice_id="mms-hin-vits",
                name="Meta MMS Hindi Neural (VITS)",
                gender="Female",
                engine="mms",
                sample_rate=16000,
                is_default=True,
                description="Meta MMS end-to-end VITS neural voice for Hindi (Devanagari)",
            ),
            VoiceInfo(
                voice_id="indicf5-hi",
                name="AI4Bharat IndicF5 Hindi",
                gender="Neutral",
                engine="indicf5",
                sample_rate=24000,
                is_default=False,
                description="AI4Bharat IndicF5 flow-matching voice",
            ),
        ],
    ),
    "bn": LanguageSpec(
        code="bn",
        iso_639_3="ben",
        english_name="Bengali",
        native_name="বাংলা",
        script_name="Bengali",
        nllb_code="ben_Beng",
        indictrans_code="ben_Beng",
        mms_tts_code="ben",
        indicf5_code="bn",
        voices=[
            VoiceInfo(
                voice_id="mms-ben-vits",
                name="Meta MMS Bengali Neural (VITS)",
                gender="Female",
                engine="mms",
                sample_rate=16000,
                is_default=True,
                description="Meta MMS end-to-end VITS neural voice for Bengali",
            ),
            VoiceInfo(
                voice_id="indicf5-bn",
                name="AI4Bharat IndicF5 Bengali",
                gender="Neutral",
                engine="indicf5",
                sample_rate=24000,
                is_default=False,
                description="AI4Bharat IndicF5 voice for Bengali",
            ),
        ],
    ),
    "mr": LanguageSpec(
        code="mr",
        iso_639_3="mar",
        english_name="Marathi",
        native_name="मराठी",
        script_name="Devanagari",
        nllb_code="mar_Deva",
        indictrans_code="mar_Deva",
        mms_tts_code="mar",
        indicf5_code="mr",
        voices=[
            VoiceInfo(
                voice_id="mms-mar-vits",
                name="Meta MMS Marathi Neural (VITS)",
                gender="Female",
                engine="mms",
                sample_rate=16000,
                is_default=True,
                description="Meta MMS end-to-end VITS neural voice for Marathi",
            ),
            VoiceInfo(
                voice_id="indicf5-mr",
                name="AI4Bharat IndicF5 Marathi",
                gender="Neutral",
                engine="indicf5",
                sample_rate=24000,
                is_default=False,
                description="AI4Bharat IndicF5 voice for Marathi",
            ),
        ],
    ),
    "ta": LanguageSpec(
        code="ta",
        iso_639_3="tam",
        english_name="Tamil",
        native_name="தமிழ்",
        script_name="Tamil",
        nllb_code="tam_Taml",
        indictrans_code="tam_Taml",
        mms_tts_code="tam",
        indicf5_code="ta",
        voices=[
            VoiceInfo(
                voice_id="mms-tam-vits",
                name="Meta MMS Tamil Neural (VITS)",
                gender="Female",
                engine="mms",
                sample_rate=16000,
                is_default=True,
                description="Meta MMS end-to-end VITS neural voice for Tamil",
            ),
            VoiceInfo(
                voice_id="indicf5-ta",
                name="AI4Bharat IndicF5 Tamil",
                gender="Neutral",
                engine="indicf5",
                sample_rate=24000,
                is_default=False,
                description="AI4Bharat IndicF5 voice for Tamil",
            ),
        ],
    ),
    "te": LanguageSpec(
        code="te",
        iso_639_3="tel",
        english_name="Telugu",
        native_name="తెలుగు",
        script_name="Telugu",
        nllb_code="tel_Telu",
        indictrans_code="tel_Telu",
        mms_tts_code="tel",
        indicf5_code="te",
        voices=[
            VoiceInfo(
                voice_id="mms-tel-vits",
                name="Meta MMS Telugu Neural (VITS)",
                gender="Female",
                engine="mms",
                sample_rate=16000,
                is_default=True,
                description="Meta MMS end-to-end VITS neural voice for Telugu",
            ),
            VoiceInfo(
                voice_id="indicf5-te",
                name="AI4Bharat IndicF5 Telugu",
                gender="Neutral",
                engine="indicf5",
                sample_rate=24000,
                is_default=False,
                description="AI4Bharat IndicF5 voice for Telugu",
            ),
        ],
    ),
    "gu": LanguageSpec(
        code="gu",
        iso_639_3="guj",
        english_name="Gujarati",
        native_name="ગુજરાતી",
        script_name="Gujarati",
        nllb_code="guj_Gujr",
        indictrans_code="guj_Gujr",
        mms_tts_code="guj",
        indicf5_code="gu",
        voices=[
            VoiceInfo(
                voice_id="mms-guj-vits",
                name="Meta MMS Gujarati Neural (VITS)",
                gender="Female",
                engine="mms",
                sample_rate=16000,
                is_default=True,
                description="Meta MMS end-to-end VITS neural voice for Gujarati",
            ),
            VoiceInfo(
                voice_id="indicf5-gu",
                name="AI4Bharat IndicF5 Gujarati",
                gender="Neutral",
                engine="indicf5",
                sample_rate=24000,
                is_default=False,
                description="AI4Bharat IndicF5 voice for Gujarati",
            ),
        ],
    ),
    "kn": LanguageSpec(
        code="kn",
        iso_639_3="kan",
        english_name="Kannada",
        native_name="ಕನ್ನಡ",
        script_name="Kannada",
        nllb_code="kan_Knda",
        indictrans_code="kan_Knda",
        mms_tts_code="kan",
        indicf5_code="kn",
        voices=[
            VoiceInfo(
                voice_id="mms-kan-vits",
                name="Meta MMS Kannada Neural (VITS)",
                gender="Female",
                engine="mms",
                sample_rate=16000,
                is_default=True,
                description="Meta MMS end-to-end VITS neural voice for Kannada",
            ),
            VoiceInfo(
                voice_id="indicf5-kn",
                name="AI4Bharat IndicF5 Kannada",
                gender="Neutral",
                engine="indicf5",
                sample_rate=24000,
                is_default=False,
                description="AI4Bharat IndicF5 voice for Kannada",
            ),
        ],
    ),
    "ml": LanguageSpec(
        code="ml",
        iso_639_3="mal",
        english_name="Malayalam",
        native_name="മലയാളം",
        script_name="Malayalam",
        nllb_code="mal_Mlym",
        indictrans_code="mal_Mlym",
        mms_tts_code="mal",
        indicf5_code="ml",
        voices=[
            VoiceInfo(
                voice_id="mms-mal-vits",
                name="Meta MMS Malayalam Neural (VITS)",
                gender="Female",
                engine="mms",
                sample_rate=16000,
                is_default=True,
                description="Meta MMS end-to-end VITS neural voice for Malayalam",
            ),
            VoiceInfo(
                voice_id="indicf5-ml",
                name="AI4Bharat IndicF5 Malayalam",
                gender="Neutral",
                engine="indicf5",
                sample_rate=24000,
                is_default=False,
                description="AI4Bharat IndicF5 voice for Malayalam",
            ),
        ],
    ),
    "pa": LanguageSpec(
        code="pa",
        iso_639_3="pan",
        english_name="Punjabi",
        native_name="ਪੰਜਾਬੀ",
        script_name="Gurmukhi",
        nllb_code="pan_Guru",
        indictrans_code="pan_Guru",
        mms_tts_code="pan",
        indicf5_code="pa",
        voices=[
            VoiceInfo(
                voice_id="mms-pan-vits",
                name="Meta MMS Punjabi Neural (VITS)",
                gender="Female",
                engine="mms",
                sample_rate=16000,
                is_default=True,
                description="Meta MMS end-to-end VITS neural voice for Punjabi",
            ),
        ],
    ),
    "or": LanguageSpec(
        code="or",
        iso_639_3="ory",
        english_name="Odia",
        native_name="ଓଡ଼ିଆ",
        script_name="Odia",
        nllb_code="ory_Orya",
        indictrans_code="ory_Orya",
        mms_tts_code="ory",
        indicf5_code="or",
        voices=[
            VoiceInfo(
                voice_id="mms-ory-vits",
                name="Meta MMS Odia Neural (VITS)",
                gender="Female",
                engine="mms",
                sample_rate=16000,
                is_default=True,
                description="Meta MMS end-to-end VITS neural voice for Odia",
            ),
        ],
    ),
    "en": LanguageSpec(
        code="en",
        iso_639_3="eng",
        english_name="English",
        native_name="English",
        script_name="Latin",
        nllb_code="eng_Latn",
        indictrans_code="eng_Latn",
        mms_tts_code="eng",
        indicf5_code="en",
        voices=[
            VoiceInfo(
                voice_id="mms-eng-vits",
                name="Meta MMS English Neural (VITS)",
                gender="Female",
                engine="mms",
                sample_rate=16000,
                is_default=True,
                description="Meta MMS end-to-end VITS neural voice for English",
            ),
        ],
    ),
}


# Index lookups
_LOOKUP_MAP: Dict[str, LanguageSpec] = {}
for spec in LANGUAGES.values():
    _LOOKUP_MAP[spec.code.lower()] = spec
    _LOOKUP_MAP[spec.iso_639_3.lower()] = spec
    _LOOKUP_MAP[spec.english_name.lower()] = spec
    _LOOKUP_MAP[spec.native_name.lower()] = spec
    _LOOKUP_MAP[spec.nllb_code.lower()] = spec
    _LOOKUP_MAP[spec.indictrans_code.lower()] = spec
    _LOOKUP_MAP[spec.mms_tts_code.lower()] = spec


def resolve_language(identifier: str) -> Optional[LanguageSpec]:
    """
    Resolve any language identifier (code, name, native script, NLLB tag)
    to its canonical LanguageSpec.
    Returns None if unsupported.
    """
    if not identifier:
        return None
    cleaned = identifier.strip().lower()
    return _LOOKUP_MAP.get(cleaned)


def get_supported_languages() -> List[Dict]:
    """Return serializable summary of all supported languages."""
    return [
        {
            "code": spec.code,
            "iso_639_3": spec.iso_639_3,
            "name": spec.english_name,
            "native_name": spec.native_name,
            "script": spec.script_name,
            "nllb_code": spec.nllb_code,
            "indictrans_code": spec.indictrans_code,
            "default_voice": next((v.voice_id for v in spec.voices if v.is_default), None),
        }
        for spec in LANGUAGES.values()
    ]


def get_all_voices() -> List[Dict]:
    """Return all registered voices across all languages."""
    all_voices = []
    for spec in LANGUAGES.values():
        for voice in spec.voices:
            all_voices.append({
                "voice_id": voice.voice_id,
                "name": voice.name,
                "gender": voice.gender,
                "engine": voice.engine,
                "sample_rate": voice.sample_rate,
                "is_default": voice.is_default,
                "description": voice.description,
                "language_code": spec.code,
                "language_name": spec.english_name,
                "native_language_name": spec.native_name,
            })
    return all_voices
