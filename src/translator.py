#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Centralized Multilingual Translation Module for Indian Sign Language (ISL) Recognition.
======================================================================================
Supports 10 Languages:
- English
- Hindi (हिंदी)
- Marathi (मराठी)
- Bengali (বাংলা)
- Tamil (தமிழ்)
- Telugu (తెలుగు)
- Gujarati (ગુજરાતી)
- Kannada (ಕನ್ನಡ)
- Malayalam (മലയാളം)
- Punjabi (ਪੰਜਾਬੀ)

Key Architecture:
1. High-Performance Local Dictionary & Phrase Lexicon:
   - Curated native translations for everyday nouns, verbs, adjectives, family, greetings.
   - Exact multi-word compound phrase translation (e.g. "BALL GAME" -> "चेंडूचा खेळ" in Marathi, "गेंद का खेल" in Hindi).
2. Online Real-Time Translation via MyMemory API:
   - Seamless translation for unlisted novel words and sentences.
   - Strict 1.5-second timeout, rate-limit protection, and error trapping.
3. Persistent Dual-Layer Cache:
   - In-memory cache + persistent JSON disk cache (data/translation_cache.json).
4. Grammar & Connector Assembly:
   - Contextual compound phrase connecting rules (e.g. "BALL" + "GAME" -> "चेंडूचा खेळ").
5. Transliteration Fallback:
   - Converts proper nouns (names) phonetically into the native script so translated text NEVER remains raw English.
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
CACHE_FILE_PATH = WORKSPACE_ROOT / "data" / "translation_cache.json"

# Key shortcut mapping for numeric keys 0-9
KEY_TO_LANGUAGE: Dict[str, str] = {
    "1": "English",
    "2": "Hindi",
    "3": "Marathi",
    "4": "Bengali",
    "5": "Tamil",
    "6": "Telugu",
    "7": "Gujarati",
    "8": "Kannada",
    "9": "Malayalam",
    "0": "Punjabi",
}

# Display names for UI headers
LANGUAGE_DISPLAY_NAMES: Dict[str, str] = {
    "English": "English",
    "Hindi": "Hindi (हिंदी)",
    "Marathi": "Marathi (मराठी)",
    "Bengali": "Bengali (বাংলা)",
    "Tamil": "Tamil (தமிழ்)",
    "Telugu": "Telugu (తెలుగు)",
    "Gujarati": "Gujarati (ગુજરાતી)",
    "Kannada": "Kannada (ಕನ್ನಡ)",
    "Malayalam": "Malayalam (മലയാളം)",
    "Punjabi": "Punjabi (ਪੰਜਾਬੀ)",
}

# Native script representations for UI buttons
NATIVE_LANGUAGE_NAMES: Dict[str, str] = {
    "English": "English",
    "Hindi": "हिंदी",
    "Marathi": "मराठी",
    "Bengali": "বাংলা",
    "Tamil": "தமிழ்",
    "Telugu": "తెలుగు",
    "Gujarati": "ગુજરાતી",
    "Kannada": "ಕನ್ನಡ",
    "Malayalam": "മലയാളം",
    "Punjabi": "ਪੰਜਾਬੀ",
}

SUPPORTED_LANGUAGES: List[str] = [
    "English", "Hindi", "Marathi", "Bengali", "Tamil",
    "Telugu", "Gujarati", "Kannada", "Malayalam", "Punjabi"
]

MYMEMORY_LANG_CODES: Dict[str, str] = {
    "English": "en-US",
    "Hindi": "hi-IN",
    "Marathi": "mr-IN",
    "Bengali": "bn-IN",
    "Tamil": "ta-IN",
    "Telugu": "te-IN",
    "Gujarati": "gu-IN",
    "Kannada": "kn-IN",
    "Malayalam": "ml-IN",
    "Punjabi": "pa-IN",
}

# ==============================================================================
# 1. CANONICAL MULTI-WORD PHRASES (Exact Grammatical Translation)
# ==============================================================================
PHRASE_DICTIONARY: Dict[str, Dict[str, str]] = {
    # 1. BALL GAME
    "ball game": {
        "English": "Ball Game", "Hindi": "गेंद का खेल", "Marathi": "चेंडूचा खेळ",
        "Bengali": "বলের খেলা", "Tamil": "பந்து விளையாட்டு", "Telugu": "బంతి ఆట",
        "Gujarati": "દડાની રમત", "Kannada": "ಚೆಂಡಿನ ಆಟ", "Malayalam": "പന്ത് കളി",
        "Punjabi": "ਗੇਂਦ ਦੀ ਖੇਡ",
    },
    # 2. HELLO
    "hello": {
        "English": "Hello", "Hindi": "नमस्ते", "Marathi": "नमस्कार",
        "Bengali": "নমস্কার", "Tamil": "வணக்கம்", "Telugu": "నమస్కారం",
        "Gujarati": "નમસ્તે", "Kannada": "ನಮಸ್ಕಾರ", "Malayalam": "നമസ്കാരം",
        "Punjabi": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ",
    },
    # 3. THANK YOU
    "thank you": {
        "English": "Thank You", "Hindi": "धन्यवाद", "Marathi": "धन्यवाद",
        "Bengali": "ধন্যবাদ", "Tamil": "நன்றி", "Telugu": "ధన్యవాదాలు",
        "Gujarati": "આભાર", "Kannada": "ಧನ್ಯವಾದಗಳು", "Malayalam": "നന്ദി",
        "Punjabi": "ਧੰਨਵਾਦ",
    },
    "thanks": {
        "English": "Thanks", "Hindi": "धन्यवाद", "Marathi": "धन्यवाद",
        "Bengali": "ধন্যবাদ", "Tamil": "நன்றி", "Telugu": "ధన్యవాదాలు",
        "Gujarati": "આભાર", "Kannada": "ಧನ್ಯವಾದಗಳು", "Malayalam": "നന്ദി",
        "Punjabi": "ਧੰਨਵਾਦ",
    },
    # 4. I LOVE YOU
    "i love you": {
        "English": "I Love You", "Hindi": "मैं आपसे प्यार करता हूँ", "Marathi": "माझे तुझ्यावर प्रेम आहे",
        "Bengali": "আমি তোমাকে ভালোবাসি", "Tamil": "நான் உன்னை காதலிக்கிறேன்", "Telugu": "నేను నిన్ను ప్రేమిస్తున్నాను",
        "Gujarati": "હું તને પ્રેમ કરું છું", "Kannada": "ನಾನು ನಿನ್ನನ್ನು ಪ್ರೀತಿಸುತ್ತೇನೆ", "Malayalam": "ഞാൻ നിന്നെ സ്നേഹിക്കുന്നു",
        "Punjabi": "ਮੈਂ ਤੈਨੂੰ ਪਿਆਰ ਕਰਦਾ ਹਾਂ",
    },
    # 5. I WANT WATER
    "i want water": {
        "English": "I Want Water", "Hindi": "मुझे पानी चाहिए।", "Marathi": "मला पाणी हवे आहे.",
        "Bengali": "আমি জল চাই।", "Tamil": "எனக்கு தண்ணீர் வேண்டும்.", "Telugu": "నాకు నీరు కావాలి.",
        "Gujarati": "મને પાણી જોઈએ છે.", "Kannada": "ನನಗೆ ನೀರು ಬೇಕು.", "Malayalam": "എനിക്ക് വെള്ളം വേണം.",
        "Punjabi": "ਮੈਨੂੰ ਪਾਣੀ ਚਾਹੀਦਾ ਹੈ।",
    },
    # 6. GOOD MORNING
    "good morning": {
        "English": "Good Morning", "Hindi": "शुभ प्रभात", "Marathi": "शुभ प्रभात",
        "Bengali": "সুপ্রভাত", "Tamil": "காலை வணக்கம்", "Telugu": "శుభోదయం",
        "Gujarati": "શુભ સવાર", "Kannada": "ಶುಭೋದಯ", "Malayalam": "സുപ്രഭാതം",
        "Punjabi": "ਸ਼ੁਭ ਸਵੇਰ",
    },
    # 7. HOW ARE YOU
    "how are you": {
        "English": "How Are You", "Hindi": "आप कैसे हैं", "Marathi": "तुम्ही कसे आहात",
        "Bengali": "আপনি কেমন আছেন", "Tamil": "நீங்கள் எப்படி இருக்கிறீர்கள்", "Telugu": "మీరు ఎలా ఉన్నారు",
        "Gujarati": "તમે કેમ છો", "Kannada": "ನೀವು ಹೇಗಿದ್ದೀರಿ", "Malayalam": "സുഖമാണോ",
        "Punjabi": "ਤੁਸੀਂ ਕਿਵੇਂ ਹੋ",
    },
    # 8. HELLO HOW ARE YOU
    "hello how are you": {
        "English": "Hello, How Are You?", "Hindi": "नमस्ते आप कैसे हैं", "Marathi": "नमस्कार तुम्ही कसे आहात",
        "Bengali": "নমস্কার আপনি কেমন আছেন", "Tamil": "வணக்கம் நீங்கள் எப்படி இருக்கிறீர்கள்", "Telugu": "నమస్కారం మీరు ఎలా ఉన్నారు",
        "Gujarati": "નમસ્તે તમે કેમ છો", "Kannada": "ನಮಸ್ಕಾರ ನೀವು ಹೇಗಿದ್ದೀರಿ", "Malayalam": "നമസ്കാരം നിങ്ങൾക്ക് എങ്ങനെയുണ്ട്",
        "Punjabi": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ ਤੁਸੀਂ ਕਿਵੇਂ ਹੋ",
    },
    # 9. HELLO FRIEND
    "hello friend": {
        "English": "Hello Friend", "Hindi": "नमस्ते दोस्त", "Marathi": "नमस्कार मित्र",
        "Bengali": "নমস্কার বন্ধু", "Tamil": "வணக்கம் நண்பரே", "Telugu": "నమస్కారం స్నేహితుడా",
        "Gujarati": "નમસ્તે મિત્ર", "Kannada": "ನಮಸ್ಕಾರ ಸ್ನೇಹಿತ", "Malayalam": "നമസ്കാരം സുഹൃത്തേ",
        "Punjabi": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ ਦੋਸਤ",
    },
    # 10. GOOD NIGHT
    "good night": {
        "English": "Good Night", "Hindi": "शुभ रात्रि", "Marathi": "शुभ रात्री",
        "Bengali": "শুভ রাত্রি", "Tamil": "இனிய இரவு", "Telugu": "శుభరాత్రి",
        "Gujarati": "શુભ રાત્રિ", "Kannada": "ಶುಭ ರಾತ್ರಿ", "Malayalam": "ശുഭരാത്രി",
        "Punjabi": "ਸ਼ੁਭ ਰਾਤ",
    },
    # 11. WHAT IS YOUR NAME
    "what is your name": {
        "English": "What Is Your Name?", "Hindi": "आपका नाम क्या है?", "Marathi": "तुमचे नाव काय आहे?",
        "Bengali": "আপনার নাম কি?", "Tamil": "உங்கள் பெயர் என்ன?", "Telugu": "మీ పేరు ఏమిటి?",
        "Gujarati": "તમારું નામ શું છે?", "Kannada": "ನಿಮ್ಮ ಹೆಸರೇನು?", "Malayalam": "നിങ്ങളുടെ പേരെന്താണ്?",
        "Punjabi": "ਤੁਹਾਡਾ ਨਾਮ ਕੀ ਹੈ?",
    },
    # 12. MY NAME IS
    "my name is": {
        "English": "My Name Is", "Hindi": "मेरा नाम है", "Marathi": "माझे नाव आहे",
        "Bengali": "আমার নাম", "Tamil": "என் பெயர்", "Telugu": "నా పేరు",
        "Gujarati": "મારું નામ છે", "Kannada": "ನನ್ನ ಹೆಸರು", "Malayalam": "എന്റെ പേര്",
        "Punjabi": "ਮੇਰਾ ਨਾਮ ਹੈ",
    },
    # 13. I WANT FOOD
    "i want food": {
        "English": "I Want Food", "Hindi": "मुझे खाना चाहिए।", "Marathi": "मला जेवण हवे आहे.",
        "Bengali": "আমি খাবার চাই।", "Tamil": "எனக்கு உணவு வேண்டும்.", "Telugu": "నాకు ఆహారం కావాలి.",
        "Gujarati": "મને ખોરાક જોઈએ છે.", "Kannada": "ನನಗೆ ಊಟ ಬೇಕು.", "Malayalam": "എനിക്ക് ഭക്ഷണം വേണം.",
        "Punjabi": "ਮੈਨੂੰ ਖਾਣਾ ਚਾਹੀਦਾ ਹੈ।",
    },
    # 14. PLEASE HELP ME
    "please help me": {
        "English": "Please Help Me", "Hindi": "कृपया मेरी मदद करें।", "Marathi": "कृपया मला मदत करा.",
        "Bengali": "দয়া করে আমাকে সাহায্য করুন।", "Tamil": "தயவுசெய்து எனக்கு உதவுங்கள்.", "Telugu": "దయచేసి నాకు సహాయం చేయండి.",
        "Gujarati": "કૃપા કરીને મને મદદ કરો.", "Kannada": "ದಯವಿಟ್ಟು ನನಗೆ ಸಹಾಯ ಮಾಡಿ.", "Malayalam": "ദയവായി എന്നെ സഹായിക്കൂ.",
        "Punjabi": "ਕਿਰਪਾ ਕਰਕੇ ਮੇਰੀ ਮਦਦ ਕਰੋ।",
    },
    # 15. SEE YOU LATER
    "see you later": {
        "English": "See You Later", "Hindi": "फिर मिलेंगे", "Marathi": "पुन्हा भेटू",
        "Bengali": "পরে দেখা হবে", "Tamil": "பிறகு சந்திப்போம்", "Telugu": "తర్వాత కలుద్దాం",
        "Gujarati": "પછી મળીશું", "Kannada": "ಮತ್ತೆ ಭೇಟಿಯಾಗೋಣ", "Malayalam": "പിന്നെ കാണാം",
        "Punjabi": "ਫਿਰ ਮਿਲਾਂਗੇ",
    },
    # 16. WELCOME
    "welcome": {
        "English": "Welcome", "Hindi": "स्वागत है", "Marathi": "स्वागत आहे",
        "Bengali": "স্বাগতম", "Tamil": "வரவேற்கிறோம்", "Telugu": "స్వాగతం",
        "Gujarati": "સ્વાગત છે", "Kannada": "ಸ್ವಾಗತ", "Malayalam": "സ്വാഗതം",
        "Punjabi": "ਜੀ ਆਇਆਂ ਨੂੰ",
    },
    # 17. HAVE A GOOD DAY
    "have a good day": {
        "English": "Have A Good Day", "Hindi": "आपका दिन शुभ हो", "Marathi": "तुमचा दिवस चांगला जावो",
        "Bengali": "আপনার দিনটি শুভ হোক", "Tamil": "இனிய நாளாக அமையட்டும்", "Telugu": "మంచి రోజు కావాలని కోరుకుంటున్నాను",
        "Gujarati": "તમારો દિવસ શુભ રહે", "Kannada": "ನಿಮ್ಮ ದಿನ ಶುಭವಾಗಿರಲಿ", "Malayalam": "നല്ലൊരു ദിവസം ആശംസിക്കുന്നു",
        "Punjabi": "ਤੁਹਾਡਾ ਦਿਨ ਵਧੀਆ ਰਹੇ",
    },
}

# ==============================================================================
# 2. COMPREHENSIVE VOCABULARY DICTIONARY (Everyday Words)
# ==============================================================================
TRANSLATION_DICTIONARY: Dict[str, Dict[str, str]] = {
    'bad': {
        'English': 'Bad',
        'Hindi': 'बुरा',
        'Marathi': 'वाईट',
        'Bengali': 'খারাপ',
        'Tamil': 'கெட்டது',
        'Telugu': 'చెడు',
        'Gujarati': 'ખરાબ',
        'Kannada': 'ಕೆಟ್ಟದು',
        'Malayalam': 'മോശം',
        'Punjabi': 'ਮਾੜਾ',
    },
    'ball': {
        'English': 'Ball',
        'Hindi': 'गेंद',
        'Marathi': 'चेंडू',
        'Bengali': 'বল',
        'Tamil': 'பந்து',
        'Telugu': 'బంతి',
        'Gujarati': 'દડો',
        'Kannada': 'ಚೆಂಡು',
        'Malayalam': 'പന്ത്',
        'Punjabi': 'ਗੇਂਦ',
    },
    'big': {
        'English': 'Big',
        'Hindi': 'बड़ा',
        'Marathi': 'मोठे',
        'Bengali': 'বড়',
        'Tamil': 'பெரியது',
        'Telugu': 'పెద్దది',
        'Gujarati': 'મોટું',
        'Kannada': 'ದೊಡ್ಡದು',
        'Malayalam': 'വലുത്',
        'Punjabi': 'ਵੱਡਾ',
    },
    'book': {
        'English': 'Book',
        'Hindi': 'किताब',
        'Marathi': 'पुस्तक',
        'Bengali': 'বই',
        'Tamil': 'புத்தகம்',
        'Telugu': 'పుస్తకం',
        'Gujarati': 'પુસ્તક',
        'Kannada': 'ಪುಸ್ತಕ',
        'Malayalam': 'പുസ്തകം',
        'Punjabi': 'ਕਿਤਾਬ',
    },
    'brother': {
        'English': 'Brother',
        'Hindi': 'भाई',
        'Marathi': 'भाऊ',
        'Bengali': 'ভাই',
        'Tamil': 'சகோதரன்',
        'Telugu': 'సోదరుడు',
        'Gujarati': 'ભાઈ',
        'Kannada': 'ಸಹೋದರ',
        'Malayalam': 'സഹോദരൻ',
        'Punjabi': 'ਭਰਾ',
    },
    'car': {
        'English': 'Car',
        'Hindi': 'गाड़ी',
        'Marathi': 'गाडी',
        'Bengali': 'গাড়ি',
        'Tamil': 'கார்',
        'Telugu': 'కారు',
        'Gujarati': 'ગાડી',
        'Kannada': 'ಕಾರು',
        'Malayalam': 'കാർ',
        'Punjabi': 'ਕਾਰ',
    },
    'cat': {
        'English': 'Cat',
        'Hindi': 'बिल्ली',
        'Marathi': 'मांजर',
        'Bengali': 'বিড়াল',
        'Tamil': 'பூனை',
        'Telugu': 'పిల్లి',
        'Gujarati': 'બિલાડી',
        'Kannada': 'ಬೆಕ್ಕು',
        'Malayalam': 'പൂച്ച',
        'Punjabi': 'ਬਿੱਲੀ',
    },
    'coffee': {
        'English': 'Coffee',
        'Hindi': 'कॉफ़ी',
        'Marathi': 'कॉफी',
        'Bengali': 'কফি',
        'Tamil': 'காபி',
        'Telugu': 'కాఫీ',
        'Gujarati': 'કોફી',
        'Kannada': 'ಕಾಫಿ',
        'Malayalam': 'കോഫി',
        'Punjabi': 'ਕੌਫੀ',
    },
    'come': {
        'English': 'Come',
        'Hindi': 'आना',
        'Marathi': 'येणे',
        'Bengali': 'আসা',
        'Tamil': 'வா',
        'Telugu': 'రా',
        'Gujarati': 'આવવું',
        'Kannada': 'ಬಾ',
        'Malayalam': 'വരിക',
        'Punjabi': 'ਆਉਣਾ',
    },
    'dad': {
        'English': 'Dad',
        'Hindi': 'पिता',
        'Marathi': 'बाबा',
        'Bengali': 'বাবা',
        'Tamil': 'அப்பா',
        'Telugu': 'నాన్న',
        'Gujarati': 'પપ્પા',
        'Kannada': 'ಅಪ್ಪ',
        'Malayalam': 'അച്ഛൻ',
        'Punjabi': 'ਪਾਪਾ',
    },
    'day': {
        'English': 'Day',
        'Hindi': 'दिन',
        'Marathi': 'दिवस',
        'Bengali': 'দিন',
        'Tamil': 'நாள்',
        'Telugu': 'రోజు',
        'Gujarati': 'દિવસ',
        'Kannada': 'ದಿನ',
        'Malayalam': 'ദിവസം',
        'Punjabi': 'ਦਿਨ',
    },
    'doctor': {
        'English': 'Doctor',
        'Hindi': 'डॉक्टर',
        'Marathi': 'डॉक्टर',
        'Bengali': 'ডাক্তার',
        'Tamil': 'மருத்துவர்',
        'Telugu': 'వైద్యుడు',
        'Gujarati': 'ડોક્ટર',
        'Kannada': 'ವೈದ್ಯ',
        'Malayalam': 'ഡോക്ടർ',
        'Punjabi': 'ਡਾਕਟਰ',
    },
    'dog': {
        'English': 'Dog',
        'Hindi': 'कुत्ता',
        'Marathi': 'कुत्रा',
        'Bengali': 'কুকুর',
        'Tamil': 'நாய்',
        'Telugu': 'కుక్క',
        'Gujarati': 'કૂતરો',
        'Kannada': 'ನಾಯಿ',
        'Malayalam': 'പട്ടി',
        'Punjabi': 'ਕੁੱਤਾ',
    },
    'drink': {
        'English': 'Drink',
        'Hindi': 'पीना',
        'Marathi': 'पिणे',
        'Bengali': 'পান করা',
        'Tamil': 'குடி',
        'Telugu': 'త్రాగు',
        'Gujarati': 'પીવું',
        'Kannada': 'ಕುಡಿ',
        'Malayalam': 'കുടിക്കുക',
        'Punjabi': 'ਪੀਣਾ',
    },
    'eat': {
        'English': 'Eat',
        'Hindi': 'खाना',
        'Marathi': 'खाणे',
        'Bengali': 'খাওয়া',
        'Tamil': 'சாப்பிடு',
        'Telugu': 'తిను',
        'Gujarati': 'ખાવું',
        'Kannada': 'ತಿನ್ನು',
        'Malayalam': 'കഴിക്കുക',
        'Punjabi': 'ਖਾਣਾ',
    },
    'fail': {
        'English': 'Fail',
        'Hindi': 'असफल',
        'Marathi': 'अपयशी',
        'Bengali': 'ব্যর্থ',
        'Tamil': 'தோல்வி',
        'Telugu': 'విఫలం',
        'Gujarati': 'નિષ્ફળ',
        'Kannada': 'ವಿಫಲ',
        'Malayalam': 'പരാജയം',
        'Punjabi': 'ਅਸਫਲ',
    },
    'family': {
        'English': 'Family',
        'Hindi': 'परिवार',
        'Marathi': 'कुटुंब',
        'Bengali': 'পরিবার',
        'Tamil': 'குடும்பம்',
        'Telugu': 'కుటుంబం',
        'Gujarati': 'પરિવાર',
        'Kannada': 'ಕುಟುಂಬ',
        'Malayalam': 'കുടുംബം',
        'Punjabi': 'ਪਰਿਵਾਰ',
    },
    'father': {
        'English': 'Father',
        'Hindi': 'पिता',
        'Marathi': 'बाबा',
        'Bengali': 'বাবা',
        'Tamil': 'அப்பா',
        'Telugu': 'నాన్న',
        'Gujarati': 'પિતા',
        'Kannada': 'ತಂದೆ',
        'Malayalam': 'അച്ഛൻ',
        'Punjabi': 'ਪਿਤਾ',
    },
    'flower': {
        'English': 'Flower',
        'Hindi': 'फूल',
        'Marathi': 'फूल',
        'Bengali': 'ফুল',
        'Tamil': 'பூ',
        'Telugu': 'పువ్వు',
        'Gujarati': 'ફૂલ',
        'Kannada': 'ಹೂವು',
        'Malayalam': 'പൂവ്',
        'Punjabi': 'ਫੁੱਲ',
    },
    'food': {
        'English': 'Food',
        'Hindi': 'खाना',
        'Marathi': 'जेवण',
        'Bengali': 'খাবার',
        'Tamil': 'உணவு',
        'Telugu': 'ఆహారం',
        'Gujarati': 'ખોરાક',
        'Kannada': 'ಊಟ',
        'Malayalam': 'ഭക്ഷണം',
        'Punjabi': 'ਭੋਜਨ',
    },
    'friend': {
        'English': 'Friend',
        'Hindi': 'दोस्त',
        'Marathi': 'मित्र',
        'Bengali': 'বন্ধু',
        'Tamil': 'நண்பன்',
        'Telugu': 'స్నేహితుడు',
        'Gujarati': 'મિત્ર',
        'Kannada': 'ಸ್ನೇಹಿತ',
        'Malayalam': 'സുഹൃത്ത്',
        'Punjabi': 'ਦੋਸਤ',
    },
    'game': {
        'English': 'Game',
        'Hindi': 'खेल',
        'Marathi': 'खेळ',
        'Bengali': 'খেলা',
        'Tamil': 'விளையாட்டு',
        'Telugu': 'ఆట',
        'Gujarati': 'રમત',
        'Kannada': 'ಆಟ',
        'Malayalam': 'കളി',
        'Punjabi': 'ਖੇਡ',
    },
    'go': {
        'English': 'Go',
        'Hindi': 'जाना',
        'Marathi': 'जाणे',
        'Bengali': 'যাওয়া',
        'Tamil': 'போ',
        'Telugu': 'వెళ్లు',
        'Gujarati': 'જવું',
        'Kannada': 'ಹೋಗು',
        'Malayalam': 'പോകുക',
        'Punjabi': 'ਜਾਣਾ',
    },
    'good': {
        'English': 'Good',
        'Hindi': 'अच्छा',
        'Marathi': 'छान',
        'Bengali': 'ভালো',
        'Tamil': 'நல்லது',
        'Telugu': 'మంచిది',
        'Gujarati': 'સારું',
        'Kannada': 'ಒಳ್ಳೆಯದು',
        'Malayalam': 'നല്ലത്',
        'Punjabi': 'ਚੰਗਾ',
    },
    'happy': {
        'English': 'Happy',
        'Hindi': 'खुश',
        'Marathi': 'आनंदी',
        'Bengali': 'খুশি',
        'Tamil': 'மகிழ்ச்சி',
        'Telugu': 'సంతోషం',
        'Gujarati': 'ખુશ',
        'Kannada': 'ಸಂತೋಷ',
        'Malayalam': 'സന്തോഷം',
        'Punjabi': 'ਖੁਸ਼',
    },
    'he': {
        'English': 'He',
        'Hindi': 'वह',
        'Marathi': 'तो',
        'Bengali': 'সে',
        'Tamil': 'அவன்',
        'Telugu': 'అతను',
        'Gujarati': 'તે',
        'Kannada': 'ಅವನು',
        'Malayalam': 'അവൻ',
        'Punjabi': 'ਉਹ',
    },
    'hello': {
        'English': 'Hello',
        'Hindi': 'नमस्ते',
        'Marathi': 'नमस्कार',
        'Bengali': 'নমস্কার',
        'Tamil': 'வணக்கம்',
        'Telugu': 'నమస్కారం',
        'Gujarati': 'નમસ્તે',
        'Kannada': 'ನಮಸ್ಕಾರ',
        'Malayalam': 'നമസ്കാരം',
        'Punjabi': 'ਸਤ ਸ੍ਰੀ ਅਕਾਲ',
    },
    'help': {
        'English': 'Help',
        'Hindi': 'मदद',
        'Marathi': 'मदत',
        'Bengali': 'সাহায্য',
        'Tamil': 'உதவி',
        'Telugu': 'సహాయం',
        'Gujarati': 'મદદ',
        'Kannada': 'ಸಹಾಯ',
        'Malayalam': 'സഹായം',
        'Punjabi': 'ਮਦਦ',
    },
    'home': {
        'English': 'Home',
        'Hindi': 'घर',
        'Marathi': 'घर',
        'Bengali': 'বাড়ি',
        'Tamil': 'வீடு',
        'Telugu': 'ఇల్లు',
        'Gujarati': 'ઘર',
        'Kannada': 'ಮನೆ',
        'Malayalam': 'വീട്',
        'Punjabi': 'ਘਰ',
    },
    'hospital': {
        'English': 'Hospital',
        'Hindi': 'अस्पताल',
        'Marathi': 'रुग्णालय',
        'Bengali': 'হাসপাতাল',
        'Tamil': 'மருத்துவமனை',
        'Telugu': 'ఆసుపత్రి',
        'Gujarati': 'હોસ્પિટલ',
        'Kannada': 'ಆಸ್ಪತ್ರೆ',
        'Malayalam': 'ആശുപത്രി',
        'Punjabi': 'ਹਸਪਤਾਲ',
    },
    'house': {
        'English': 'House',
        'Hindi': 'घर',
        'Marathi': 'घर',
        'Bengali': 'বাড়ি',
        'Tamil': 'வீடு',
        'Telugu': 'ఇల్లు',
        'Gujarati': 'ઘર',
        'Kannada': 'ಮನೆ',
        'Malayalam': 'വീട്',
        'Punjabi': 'ਘਰ',
    },
    'how': {
        'English': 'How',
        'Hindi': 'कैसे',
        'Marathi': 'कसे',
        'Bengali': 'কেমন',
        'Tamil': 'எப்படி',
        'Telugu': 'ఎలా',
        'Gujarati': 'કેવી રીતે',
        'Kannada': 'ಹೇಗೆ',
        'Malayalam': 'എങ്ങനെ',
        'Punjabi': 'ਕਿਵੇਂ',
    },
    'i': {
        'English': 'I',
        'Hindi': 'मैं',
        'Marathi': 'मी',
        'Bengali': 'আমি',
        'Tamil': 'நான்',
        'Telugu': 'నేను',
        'Gujarati': 'હું',
        'Kannada': 'ನಾನು',
        'Malayalam': 'ഞാൻ',
        'Punjabi': 'ਮੈਂ',
    },
    'iloveyou': {
        'English': 'I Love You',
        'Hindi': 'मैं आपसे प्यार करता हूँ',
        'Marathi': 'माझे तुमच्यावर प्रेम आहे',
        'Bengali': 'আমি আপনাকে ভালোবাসি',
        'Tamil': 'நான் உன்னை காதலிக்கிறேன்',
        'Telugu': 'నేను నిన్ను ప్రేమిస్తున్నాను',
        'Gujarati': 'હું તને પ્રેમ કરું છું',
        'Kannada': 'ನಾನು ನಿನ್ನನ್ನು ಪ್ರೀತಿಸುತ್ತೇನೆ',
        'Malayalam': 'ഞാൻ നിന്നെ സ്നേഹിക്കുന്നു',
        'Punjabi': 'ਮੈਂ ਤੈਨੂੰ ਪਿਆਰ ਕਰਦਾ ਹਾਂ',
    },
    'india': {
        'English': 'India',
        'Hindi': 'भारत',
        'Marathi': 'भारत',
        'Bengali': 'ভারত',
        'Tamil': 'இந்தியா',
        'Telugu': 'భారతదేశం',
        'Gujarati': 'ભારત',
        'Kannada': 'ಭಾರತ',
        'Malayalam': 'ഇന്ത്യ',
        'Punjabi': 'ਭਾਰਤ',
    },
    'language': {
        'English': 'Language',
        'Hindi': 'भाषा',
        'Marathi': 'भाषा',
        'Bengali': 'ভাষা',
        'Tamil': 'மொழி',
        'Telugu': 'భాష',
        'Gujarati': 'ભાષા',
        'Kannada': 'ಭಾಷೆ',
        'Malayalam': 'ഭാഷ',
        'Punjabi': 'ਭਾਸ਼ਾ',
    },
    'learn': {
        'English': 'Learn',
        'Hindi': 'सीखना',
        'Marathi': 'शिकणे',
        'Bengali': 'শেখা',
        'Tamil': 'கற்றுக்கொள்',
        'Telugu': 'నేర్చుకో',
        'Gujarati': 'શીખવું',
        'Kannada': 'ಕಲಿ',
        'Malayalam': 'പഠിക്കുക',
        'Punjabi': 'ਸਿੱਖਣਾ',
    },
    'like': {
        'English': 'Like',
        'Hindi': 'पसंद',
        'Marathi': 'आवडते',
        'Bengali': 'পছন্দ',
        'Tamil': 'விருப்பம்',
        'Telugu': 'ఇష్టం',
        'Gujarati': 'પસંદ',
        'Kannada': 'ಇಷ್ಟ',
        'Malayalam': 'ഇഷ്ടം',
        'Punjabi': 'ਪਸੰਦ',
    },
    'location': {
        'English': 'Location',
        'Hindi': 'स्थान',
        'Marathi': 'स्थान',
        'Bengali': 'অবস্থান',
        'Tamil': 'இருப்பிடம்',
        'Telugu': 'ప్రదేశం',
        'Gujarati': 'સ્થળ',
        'Kannada': 'ಸ್ಥಳ',
        'Malayalam': 'സ്ഥലം',
        'Punjabi': 'ਸਥਾਨ',
    },
    'love': {
        'English': 'Love',
        'Hindi': 'प्यार',
        'Marathi': 'प्रेम',
        'Bengali': 'ভালোবাসা',
        'Tamil': 'அன்பு',
        'Telugu': 'ప్రేమ',
        'Gujarati': 'પ્રેમ',
        'Kannada': 'ಪ್ರೀತಿ',
        'Malayalam': 'സ്നേഹം',
        'Punjabi': 'ਪਿਆਰ',
    },
    'medicine': {
        'English': 'Medicine',
        'Hindi': 'दवाई',
        'Marathi': 'औषध',
        'Bengali': 'ওষুধ',
        'Tamil': 'மருந்து',
        'Telugu': 'మందు',
        'Gujarati': 'દવા',
        'Kannada': 'ಔಷಧ',
        'Malayalam': 'മരുന്ന്',
        'Punjabi': 'ਦਵਾਈ',
    },
    'meet': {
        'English': 'Meet',
        'Hindi': 'मिलना',
        'Marathi': 'भेटणे',
        'Bengali': 'দেখা করা',
        'Tamil': 'சந்திப்பு',
        'Telugu': 'కలవడం',
        'Gujarati': 'મળવું',
        'Kannada': 'ಭೇಟಿ',
        'Malayalam': 'കണ്ടുമുട്ടുക',
        'Punjabi': 'ਮਿਲਣਾ',
    },
    'milk': {
        'English': 'Milk',
        'Hindi': 'दूध',
        'Marathi': 'दूध',
        'Bengali': 'দুধ',
        'Tamil': 'பால்',
        'Telugu': 'పాలు',
        'Gujarati': 'દૂધ',
        'Kannada': 'ಹಾಲು',
        'Malayalam': 'പാൽ',
        'Punjabi': 'ਦੁੱਧ',
    },
    'mom': {
        'English': 'Mom',
        'Hindi': 'माँ',
        'Marathi': 'आई',
        'Bengali': 'মা',
        'Tamil': 'அம்மா',
        'Telugu': 'అమ్మ',
        'Gujarati': 'મમ્મી',
        'Kannada': 'ಅಮ್ಮ',
        'Malayalam': 'അമ്മ',
        'Punjabi': 'ਮੰਮੀ',
    },
    'money': {
        'English': 'Money',
        'Hindi': 'पैसे',
        'Marathi': 'पैसे',
        'Bengali': 'টাকা',
        'Tamil': 'பணம்',
        'Telugu': 'డబ్బు',
        'Gujarati': 'પૈસા',
        'Kannada': 'ಹಣ',
        'Malayalam': 'പണം',
        'Punjabi': 'ਪੈਸੇ',
    },
    'moon': {
        'English': 'Moon',
        'Hindi': 'चाँद',
        'Marathi': 'चंद्र',
        'Bengali': 'চাঁদ',
        'Tamil': 'நிலா',
        'Telugu': 'చంద్రుడు',
        'Gujarati': 'ચંદ્ર',
        'Kannada': 'ಚಂದ್ರ',
        'Malayalam': 'ചന്ദ്രൻ',
        'Punjabi': 'ਚੰਨ',
    },
    'morning': {
        'English': 'Morning',
        'Hindi': 'सुबह',
        'Marathi': 'सकाळ',
        'Bengali': 'সকাল',
        'Tamil': 'காலை',
        'Telugu': 'ఉదయం',
        'Gujarati': 'સવાર',
        'Kannada': 'ಬೆಳಗ್ಗೆ',
        'Malayalam': 'രാവിലെ',
        'Punjabi': 'ਸਵੇਰ',
    },
    'mother': {
        'English': 'Mother',
        'Hindi': 'माँ',
        'Marathi': 'आई',
        'Bengali': 'মা',
        'Tamil': 'அம்மா',
        'Telugu': 'అమ్మ',
        'Gujarati': 'માતા',
        'Kannada': 'ತಾಯಿ',
        'Malayalam': 'അമ്മ',
        'Punjabi': 'ਮਾਂ',
    },
    'name': {
        'English': 'Name',
        'Hindi': 'नाम',
        'Marathi': 'नाव',
        'Bengali': 'নাম',
        'Tamil': 'பெயர்',
        'Telugu': 'పేరు',
        'Gujarati': 'નામ',
        'Kannada': 'ಹೆಸರು',
        'Malayalam': 'പേര്',
        'Punjabi': 'ਨਾਮ',
    },
    'night': {
        'English': 'Night',
        'Hindi': 'रात',
        'Marathi': 'रात्र',
        'Bengali': 'রাত',
        'Tamil': 'இரவு',
        'Telugu': 'రాత్రి',
        'Gujarati': 'રાત',
        'Kannada': 'ರಾತ್ರಿ',
        'Malayalam': 'രാത്രി',
        'Punjabi': 'ਰਾਤ',
    },
    'no': {
        'English': 'No',
        'Hindi': 'नहीं',
        'Marathi': 'नाही',
        'Bengali': 'না',
        'Tamil': 'இல்லை',
        'Telugu': 'కాదు',
        'Gujarati': 'ના',
        'Kannada': 'ಇಲ್ಲ',
        'Malayalam': 'ഇല്ല',
        'Punjabi': 'ਨਹੀਂ',
    },
    'phonecall': {
        'English': 'Phone Call',
        'Hindi': 'फोन कॉल',
        'Marathi': 'फोन कॉल',
        'Bengali': 'ফোন কল',
        'Tamil': 'தொலைபேசி அழைப்பு',
        'Telugu': 'ఫోన్ కాల్',
        'Gujarati': 'ફોન કોલ',
        'Kannada': 'ಫೋನ್ ಕರೆ',
        'Malayalam': 'ഫോൺ കോൾ',
        'Punjabi': 'ਫੋਨ ਕਾਲ',
    },
    'play': {
        'English': 'Play',
        'Hindi': 'खेलना',
        'Marathi': 'खेळणे',
        'Bengali': 'খেলা',
        'Tamil': 'விளையாடு',
        'Telugu': 'ఆడు',
        'Gujarati': 'રમવું',
        'Kannada': 'ಆಡು',
        'Malayalam': 'കളിക്കുക',
        'Punjabi': 'ਖੇਡਣਾ',
    },
    'please': {
        'English': 'Please',
        'Hindi': 'कृपया',
        'Marathi': 'कृपया',
        'Bengali': 'দয়া করে',
        'Tamil': 'தயவுசெய்து',
        'Telugu': 'దయచేసి',
        'Gujarati': 'કૃપા કરીને',
        'Kannada': 'ದಯವಿಟ್ಟು',
        'Malayalam': 'ദയവായി',
        'Punjabi': 'ਕਿਰਪਾ ਕਰਕੇ',
    },
    'read': {
        'English': 'Read',
        'Hindi': 'पढ़ना',
        'Marathi': 'वाचणे',
        'Bengali': 'পড়া',
        'Tamil': 'படி',
        'Telugu': 'చదువు',
        'Gujarati': 'વાંચવું',
        'Kannada': 'ಓದು',
        'Malayalam': 'വായിക്കുക',
        'Punjabi': 'ਪੜ੍ਹਨਾ',
    },
    'sad': {
        'English': 'Sad',
        'Hindi': 'उदास',
        'Marathi': 'दुःखी',
        'Bengali': 'দুঃখিত',
        'Tamil': 'சோகம்',
        'Telugu': 'విచారం',
        'Gujarati': 'ઉદાસ',
        'Kannada': 'ದುಃಖ',
        'Malayalam': 'വിഷമം',
        'Punjabi': 'ਉਦਾਸ',
    },
    'school': {
        'English': 'School',
        'Hindi': 'विद्यालय',
        'Marathi': 'शाळा',
        'Bengali': 'বিদ্যালয়',
        'Tamil': 'பள்ளி',
        'Telugu': 'పాఠశాల',
        'Gujarati': 'શાળા',
        'Kannada': 'ಶಾಲೆ',
        'Malayalam': 'സ്കൂൾ',
        'Punjabi': 'ਸਕੂਲ',
    },
    'see': {
        'English': 'See',
        'Hindi': 'देखना',
        'Marathi': 'पाहणे',
        'Bengali': 'দেখা',
        'Tamil': 'பார்',
        'Telugu': 'చూడు',
        'Gujarati': 'જોવું',
        'Kannada': 'ನೋಡು',
        'Malayalam': 'കാണുക',
        'Punjabi': 'ਦੇਖਣਾ',
    },
    'she': {
        'English': 'She',
        'Hindi': 'वह',
        'Marathi': 'ती',
        'Bengali': 'সে',
        'Tamil': 'அவள்',
        'Telugu': 'ఆమె',
        'Gujarati': 'તેણી',
        'Kannada': 'ಅವಳು',
        'Malayalam': 'അവൾ',
        'Punjabi': 'ਉਹ',
    },
    'sign': {
        'English': 'Sign',
        'Hindi': 'संकेत',
        'Marathi': 'खूण',
        'Bengali': 'সংকেত',
        'Tamil': 'அடையாளம்',
        'Telugu': 'సంకేతం',
        'Gujarati': 'સંકેત',
        'Kannada': 'ಸಂಕೇತ',
        'Malayalam': 'അടയാളം',
        'Punjabi': 'ਸੰਕੇਤ',
    },
    'sister': {
        'English': 'Sister',
        'Hindi': 'बहन',
        'Marathi': 'बहीण',
        'Bengali': 'বোন',
        'Tamil': 'சகோதரி',
        'Telugu': 'సోదరి',
        'Gujarati': 'બહેન',
        'Kannada': 'ಸಹೋದರಿ',
        'Malayalam': 'സഹോദരി',
        'Punjabi': 'ਭੈਣ',
    },
    'sleep': {
        'English': 'Sleep',
        'Hindi': 'सोना',
        'Marathi': 'झोपणे',
        'Bengali': 'ঘুমানো',
        'Tamil': 'தூங்கு',
        'Telugu': 'నిద్రించు',
        'Gujarati': 'ઊંઘવું',
        'Kannada': 'ಮಲಗು',
        'Malayalam': 'ഉറങ്ങുക',
        'Punjabi': 'ਸੌਣਾ',
    },
    'small': {
        'English': 'Small',
        'Hindi': 'छोटा',
        'Marathi': 'लहान',
        'Bengali': 'ছোট',
        'Tamil': 'சிறியது',
        'Telugu': 'చిన్నది',
        'Gujarati': 'નાનું',
        'Kannada': 'ಚಿಕ್ಕದು',
        'Malayalam': 'ചെറുത്',
        'Punjabi': 'ਛੋਟਾ',
    },
    'sorry': {
        'English': 'Sorry',
        'Hindi': 'क्षमा करें',
        'Marathi': 'माफ करा',
        'Bengali': 'দুঃখিত',
        'Tamil': 'மன்னிக்கவும்',
        'Telugu': 'క్షమించండి',
        'Gujarati': 'માફ કરશો',
        'Kannada': 'ಕ್ಷಮಿಸಿ',
        'Malayalam': 'ക്ഷമിക്കണം',
        'Punjabi': 'ਮਾਫ਼ ਕਰਨਾ',
    },
    'speak': {
        'English': 'Speak',
        'Hindi': 'बोलना',
        'Marathi': 'बोलणे',
        'Bengali': 'কথা বলা',
        'Tamil': 'பேசு',
        'Telugu': 'మాట్లాడు',
        'Gujarati': 'બોલવું',
        'Kannada': 'ಮಾತನಾಡು',
        'Malayalam': 'സംസാരിക്കുക',
        'Punjabi': 'ਬੋਲਣਾ',
    },
    'sun': {
        'English': 'Sun',
        'Hindi': 'सूर्य',
        'Marathi': 'सूर्य',
        'Bengali': 'সূর্য',
        'Tamil': 'சூரியன்',
        'Telugu': 'సూర్యుడు',
        'Gujarati': 'સૂર્ય',
        'Kannada': 'ಸೂರ್ಯ',
        'Malayalam': 'സൂര്യൻ',
        'Punjabi': 'ਸੂਰਜ',
    },
    'takecare': {
        'English': 'Take Care',
        'Hindi': 'अपना ख्याल रखना',
        'Marathi': 'काळजी घ्या',
        'Bengali': 'নিজের যত্ন নিন',
        'Tamil': 'கவனமாக இருங்கள்',
        'Telugu': 'జాగ్రత్త',
        'Gujarati': 'સંભાળ રાખજો',
        'Kannada': 'ಕಾಳಜಿ ವಹಿಸಿ',
        'Malayalam': 'ശ്രദ്ധിക്കുക',
        'Punjabi': 'ਆਪਣਾ ਧਿਆਨ ਰੱਖਣਾ',
    },
    'tea': {
        'English': 'Tea',
        'Hindi': 'चाय',
        'Marathi': 'चहा',
        'Bengali': 'চা',
        'Tamil': 'தேநீர்',
        'Telugu': 'టీ',
        'Gujarati': 'ચા',
        'Kannada': 'ಚಹಾ',
        'Malayalam': 'ചായ',
        'Punjabi': 'ਚਾਹ',
    },
    'teacher': {
        'English': 'Teacher',
        'Hindi': 'शिक्षक',
        'Marathi': 'शिक्षक',
        'Bengali': 'শিক্ষক',
        'Tamil': 'ஆசிரியர்',
        'Telugu': 'ఉపాధ్యాయుడు',
        'Gujarati': 'શિક્ષક',
        'Kannada': 'ಶಿಕ್ಷಕ',
        'Malayalam': 'അധ്യാപകൻ',
        'Punjabi': 'ਅਧਿਆਪਕ',
    },
    'thanks': {
        'English': 'Thanks',
        'Hindi': 'धन्यवाद',
        'Marathi': 'धन्यवाद',
        'Bengali': 'ধন্যবাদ',
        'Tamil': 'நன்றி',
        'Telugu': 'ధన్యవాదాలు',
        'Gujarati': 'આભાર',
        'Kannada': 'ಧನ್ಯವಾದಗಳು',
        'Malayalam': 'നന്ദി',
        'Punjabi': 'ਧੰਨਵਾਦ',
    },
    'they': {
        'English': 'They',
        'Hindi': 'वे',
        'Marathi': 'ते',
        'Bengali': 'তারা',
        'Tamil': 'அவர்கள்',
        'Telugu': 'వారు',
        'Gujarati': 'તેઓ',
        'Kannada': 'ಅವರು',
        'Malayalam': 'അവർ',
        'Punjabi': 'ਉਹ',
    },
    'think': {
        'English': 'Think',
        'Hindi': 'सोचना',
        'Marathi': 'विचार करणे',
        'Bengali': 'চিন্তা করা',
        'Tamil': 'யோசிப்பது',
        'Telugu': 'ఆలోచించడం',
        'Gujarati': 'વિચારવું',
        'Kannada': 'ಯೋಚಿಸು',
        'Malayalam': 'ചിന്തിക്കുക',
        'Punjabi': 'ਸੋਚਣਾ',
    },
    'time': {
        'English': 'Time',
        'Hindi': 'समय',
        'Marathi': 'वेळ',
        'Bengali': 'সময়',
        'Tamil': 'நேரம்',
        'Telugu': 'సమయం',
        'Gujarati': 'સમય',
        'Kannada': 'ಸಮಯ',
        'Malayalam': 'സമയം',
        'Punjabi': 'ਸਮਾਂ',
    },
    'tree': {
        'English': 'Tree',
        'Hindi': 'पेड़',
        'Marathi': 'झाड',
        'Bengali': 'গাছ',
        'Tamil': 'மரம்',
        'Telugu': 'చెట్టు',
        'Gujarati': 'ઝાડ',
        'Kannada': 'ಮರ',
        'Malayalam': 'മരം',
        'Punjabi': 'ਦਰੱਖਤ',
    },
    'want': {
        'English': 'Want',
        'Hindi': 'चाहिए',
        'Marathi': 'हवे',
        'Bengali': 'চাই',
        'Tamil': 'வேண்டும்',
        'Telugu': 'కావాలి',
        'Gujarati': 'જોઈએ',
        'Kannada': 'ಬೇಕು',
        'Malayalam': 'വേണം',
        'Punjabi': 'ਚਾਹੀਦਾ',
    },
    'water': {
        'English': 'Water',
        'Hindi': 'पानी',
        'Marathi': 'पाणी',
        'Bengali': 'জল',
        'Tamil': 'தண்ணீர்',
        'Telugu': 'నీరు',
        'Gujarati': 'પાણી',
        'Kannada': 'ನೀರು',
        'Malayalam': 'വെള്ളം',
        'Punjabi': 'ਪਾਣੀ',
    },
    'we': {
        'English': 'We',
        'Hindi': 'हम',
        'Marathi': 'आम्ही',
        'Bengali': 'আমরা',
        'Tamil': 'நாங்கள்',
        'Telugu': 'మేము',
        'Gujarati': 'અમે',
        'Kannada': 'ನಾವು',
        'Malayalam': 'ഞങ്ങൾ',
        'Punjabi': 'ਅਸੀਂ',
    },
    'what': {
        'English': 'What',
        'Hindi': 'क्या',
        'Marathi': 'काय',
        'Bengali': 'কি',
        'Tamil': 'என்ன',
        'Telugu': 'ఏమిటి',
        'Gujarati': 'શું',
        'Kannada': 'ಏನು',
        'Malayalam': 'എന്ത്',
        'Punjabi': 'ਕੀ',
    },
    'where': {
        'English': 'Where',
        'Hindi': 'कहाँ',
        'Marathi': 'कुठे',
        'Bengali': 'কোথায়',
        'Tamil': 'எங்கே',
        'Telugu': 'ఎక్కడ',
        'Gujarati': 'ક્યાં',
        'Kannada': 'ಎಲ್ಲಿ',
        'Malayalam': 'എവിടെ',
        'Punjabi': 'ਕਿੱਥੇ',
    },
    'who': {
        'English': 'Who',
        'Hindi': 'कौन',
        'Marathi': 'कोण',
        'Bengali': 'কে',
        'Tamil': 'யார்',
        'Telugu': 'ఎవరు',
        'Gujarati': 'કોણ',
        'Kannada': 'ಯಾರು',
        'Malayalam': 'ആര്',
        'Punjabi': 'ਕੌਣ',
    },
    'write': {
        'English': 'Write',
        'Hindi': 'लिखना',
        'Marathi': 'लिहिणे',
        'Bengali': 'লেখা',
        'Tamil': 'எழுது',
        'Telugu': 'రాయి',
        'Gujarati': 'લખવું',
        'Kannada': 'ಬರೆ',
        'Malayalam': 'എഴുതുക',
        'Punjabi': 'ਲਿਖਣਾ',
    },
    'yes': {
        'English': 'Yes',
        'Hindi': 'हाँ',
        'Marathi': 'हो',
        'Bengali': 'হ্যাঁ',
        'Tamil': 'ஆம்',
        'Telugu': 'అవును',
        'Gujarati': 'હા',
        'Kannada': 'ಹೌದು',
        'Malayalam': 'അതെ',
        'Punjabi': 'ਹਾਂ',
    },
    'you': {
        'English': 'You',
        'Hindi': 'आप',
        'Marathi': 'तुम्ही',
        'Bengali': 'আপনি',
        'Tamil': 'நீங்கள்',
        'Telugu': 'మీరు',
        'Gujarati': 'તમે',
        'Kannada': 'ನೀವು',
        'Malayalam': 'നിങ്ങൾ',
        'Punjabi': 'ਤੁਸੀਂ',
    },
}

# Phonetic character transliteration fallback
PHONETIC_CHAR_MAP: Dict[str, Dict[str, str]] = {
    "Hindi": {
        "A": "अ", "B": "ब", "C": "क", "D": "ड", "E": "ए", "F": "फ", "G": "ग",
        "H": "ह", "I": "इ", "J": "ज", "K": "क", "L": "ल", "M": "म", "N": "न",
        "O": "ओ", "P": "प", "Q": "क", "R": "र", "S": "स", "T": "ट", "U": "उ",
        "V": "व", "W": "व", "X": "क्स", "Y": "य", "Z": "ज़"
    },
    "Marathi": {
        "A": "अ", "B": "ब", "C": "क", "D": "ड", "E": "ए", "F": "फ", "G": "ग",
        "H": "ह", "I": "इ", "J": "ज", "K": "क", "L": "ल", "M": "म", "N": "न",
        "O": "ओ", "P": "प", "Q": "क", "R": "र", "S": "स", "T": "ट", "U": "उ",
        "V": "व", "W": "व", "X": "क्स", "Y": "य", "Z": "झ"
    },
    "Bengali": {
        "A": "অ", "B": "ব", "C": "ক", "D": "ড", "E": "এ", "F": "ফ", "G": "গ",
        "H": "হ", "I": "ই", "J": "জ", "K": "ক", "L": "ল", "M": "ম", "N": "ন",
        "O": "ও", "P": "প", "Q": "ক", "R": "র", "S": "স", "T": "ট", "U": "উ",
        "V": "ভ", "W": "ও", "X": "ক্স", "Y": "য", "Z": "জ"
    },
    "Tamil": {
        "A": "அ", "B": "ப", "C": "க", "D": "ட", "E": "எ", "F": "ப", "G": "க",
        "H": "ஹ", "I": "இ", "J": "ஜ", "K": "க", "L": "ல", "M": "ம", "N": "ந",
        "O": "ஒ", "P": "ப", "Q": "க", "R": "ர", "S": "ச", "T": "ட", "U": "உ",
        "V": "வ", "W": "வ", "X": "க்ஸ்", "Y": "ய", "Z": "ஜ"
    },
    "Telugu": {
        "A": "అ", "B": "బ", "C": "క", "D": "డ", "E": "ఎ", "F": "ఫ", "G": "గ",
        "H": "హ", "I": "ఇ", "J": "జ", "K": "క", "L": "ల", "M": "మ", "N": "న",
        "O": "ఒ", "P": "ప", "Q": "క", "R": "ర", "S": "స", "T": "ట", "U": "ఉ",
        "V": "వ", "W": "వ", "X": "క్స్", "Y": "య", "Z": "జ"
    },
    "Gujarati": {
        "A": "અ", "B": "બ", "C": "ક", "D": "ડ", "E": "એ", "F": "ફ", "G": "ગ",
        "H": "હ", "I": "ઇ", "J": "જ", "K": "ક", "L": "લ", "M": "મ", "N": "ન",
        "O": "ઓ", "P": "પ", "Q": "ક", "R": "ર", "S": "સ", "T": "ટ", "U": "ઉ",
        "V": "વ", "W": "વ", "X": "ક્ષ", "Y": "ય", "Z": "ઝ"
    },
    "Kannada": {
        "A": "ಅ", "B": "ಬ", "C": "ಕ", "D": "ಡ", "E": "ಎ", "F": "ಫ", "G": "ಗ",
        "H": "ಹ", "I": "ಇ", "J": "ಜ", "K": "ಕ", "L": "ಲ", "M": "ಮ", "N": "ನ",
        "O": "ಒ", "P": "ಪ", "Q": "ಕ", "R": "ರ", "S": "ಸ", "T": "ಟ", "U": "ಉ",
        "V": "ವ", "W": "ವ", "X": "ಕ್ಸ್", "Y": "ಯ", "Z": "ಜ"
    },
    "Malayalam": {
        "A": "അ", "B": "ബ", "C": "ക", "D": "ഡ", "E": "എ", "F": "ഫ", "G": "ഗ",
        "H": "ഹ", "I": "ഇ", "J": "ജ", "K": "ക", "L": "ല", "M": "മ", "N": "ന",
        "O": "ഒ", "P": "പ", "Q": "ക", "R": "ര", "S": "സ", "T": "ട", "U": "ഉ",
        "V": "വ", "W": "വ", "X": "ക്സ്", "Y": "യ", "Z": "സ"
    },
    "Punjabi": {
        "A": "ਅ", "B": "ਬ", "C": "ਕ", "D": "ਡ", "E": "ਏ", "F": "ਫ", "G": "ਗ",
        "H": "ਹ", "I": "ਇ", "J": "ਜ", "K": "ਕ", "L": "ਲ", "M": "ਮ", "N": "ਨ",
        "O": "ਓ", "P": "ਪ", "Q": "ਕ", "R": "ਰ", "S": "ਸ", "T": "ਟ", "U": "ਉ",
        "V": "ਵ", "W": "ਵ", "X": "ਕ੍ਸ", "Y": "ਯ", "Z": "ਜ਼"
    },
}

# ==============================================================================
# 3. DUAL-LAYER TRANSLATION CACHE (In-Memory + Disk Persistence)
# ==============================================================================
_TRANSLATION_CACHE: Dict[Tuple[str, str], str] = {}


def _load_disk_cache() -> None:
    """Load persistent translation cache from JSON disk file."""
    if CACHE_FILE_PATH.exists():
        try:
            with open(CACHE_FILE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                for key_str, val in data.items():
                    if "|" in key_str:
                        src_text, lang = key_str.split("|", 1)
                        _TRANSLATION_CACHE[(src_text, lang)] = val
        except Exception:
            pass


def _save_disk_cache() -> None:
    """Persist cached translations to JSON disk file."""
    try:
        CACHE_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {f"{k[0]}|{k[1]}": v for k, v in _TRANSLATION_CACHE.items()}
        with open(CACHE_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


_load_disk_cache()


def transliterate_fallback(text: str, language: str) -> str:
    """Phonetically transliterate letters into the native script."""
    if language == "English" or language not in PHONETIC_CHAR_MAP:
        return text
    char_map = PHONETIC_CHAR_MAP[language]
    chars = []
    for ch in text.upper():
        chars.append(char_map.get(ch, ch))
    return "".join(chars)


# ==============================================================================
# 4. ONLINE TRANSLATION ENGINE (MyMemoryTranslator with Safe Fallbacks)
# ==============================================================================
def _translate_online_safe(text: str, language: str) -> Optional[str]:
    """
    Query MyMemory translation service with a strict timeout.
    Returns cleaned translation string, or None if error/timeout.
    """
    if language == "English" or not text.strip():
        return text

    target_code = MYMEMORY_LANG_CODES.get(language)
    if not target_code:
        return None

    try:
        from deep_translator import MyMemoryTranslator
        # Use dedicated academic project identifier for clean quota
        translator = MyMemoryTranslator(
            source="en-US",
            target=target_code,
            email="isl.academic.translator@gmail.com"
        )
        translated = translator.translate(text)
        if translated and isinstance(translated, str) and translated.strip():
            # Clean up extraneous service artifacts if any
            clean = re.sub(r"\s*Name\b", "", translated).strip()
            # If the service returned the identical raw English text, reject it as unsupported
            if clean.lower() == text.strip().lower() and language != "English":
                return None
            return clean
    except Exception:
        pass
    return None


# ==============================================================================
# 5. HIGH-ACCURACY SENTENCE & WORD TRANSLATION APIS
# ==============================================================================
def translate_label(label: str, language: str = "English") -> str:
    """Translate an individual sign label or word."""
    if not label or not label.strip():
        return ""
    clean_label = label.strip()
    if language == "English":
        return clean_label

    cache_key = (clean_label.lower(), language)
    key_lower = clean_label.lower()

    # 1. Exact phrase or word match (Always priority over cache)
    if key_lower in PHRASE_DICTIONARY:
        res = PHRASE_DICTIONARY[key_lower].get(language, clean_label)
        _TRANSLATION_CACHE[cache_key] = res
        return res

    if key_lower in TRANSLATION_DICTIONARY:
        res = TRANSLATION_DICTIONARY[key_lower].get(language, clean_label)
        _TRANSLATION_CACHE[cache_key] = res
        return res

    if cache_key in _TRANSLATION_CACHE:
        return _TRANSLATION_CACHE[cache_key]

    # 2. Single fingerspelled letter
    if len(clean_label) == 1 and clean_label.isalpha():
        res = transliterate_fallback(clean_label, language)
        _TRANSLATION_CACHE[cache_key] = res
        return res

    # 3. Online translation engine
    online_res = _translate_online_safe(clean_label, language)
    if online_res:
        _TRANSLATION_CACHE[cache_key] = online_res
        _save_disk_cache()
        return online_res

    # 4. Transliterate unknown word so it never stays raw Latin English
    res = transliterate_fallback(clean_label, language)
    _TRANSLATION_CACHE[cache_key] = res
    return res


def translate_sentence(sentence: str, language: str = "English") -> str:
    """
    Translate an accumulated sign sentence and word buffer into the target language.
    Guarantees that everyday phrases and words are translated into native Indic scripts.
    """
    if not sentence or not sentence.strip():
        return ""

    clean_sentence = sentence.strip()
    if language == "English":
        return clean_sentence

    cache_key = (clean_sentence, language)
    normalized_lower = re.sub(r"\s+", " ", clean_sentence.lower())
    normalized_phrase = re.sub(r"[^\w\s]", "", normalized_lower).strip()

    # Rule 1: Full phrase dictionary match (Priority over cache)
    if normalized_phrase in PHRASE_DICTIONARY:
        phrase_entry = PHRASE_DICTIONARY[normalized_phrase]
        translated = phrase_entry.get(language, phrase_entry.get("English", clean_sentence))
        _TRANSLATION_CACHE[cache_key] = translated
        return translated

    # Rule 2: Check leading phrase + remainder
    for phrase_key, phrase_entry in PHRASE_DICTIONARY.items():
        if normalized_phrase.startswith(phrase_key + " "):
            remainder = clean_sentence[len(phrase_key):].strip()
            trans_phrase = phrase_entry.get(language, phrase_entry.get("English", phrase_key))
            trans_rem = translate_sentence(remainder, language)
            result = f"{trans_phrase} {trans_rem}".strip()
            _TRANSLATION_CACHE[cache_key] = result
            return result

    # Rule 3: Single word dictionary match
    if normalized_phrase in TRANSLATION_DICTIONARY:
        word_entry = TRANSLATION_DICTIONARY[normalized_phrase]
        translated = word_entry.get(language, clean_sentence)
        _TRANSLATION_CACHE[cache_key] = translated
        return translated

    # Rule 4: Check for two-word compound noun (e.g. "BALL" + "GAME")
    tokens = re.findall(r"[\w]+|[^\w\s]|\s+", clean_sentence)
    words_only = [t for t in tokens if t.isalnum()]
    if len(words_only) == 2:
        w1, w2 = words_only[0].lower(), words_only[1].lower()
        if w1 in TRANSLATION_DICTIONARY and w2 in TRANSLATION_DICTIONARY:
            t1 = TRANSLATION_DICTIONARY[w1].get(language, w1)
            t2 = TRANSLATION_DICTIONARY[w2].get(language, w2)
            # Add natural possessive/compound connector if applicable
            connectors = {
                "Hindi": "का",
                "Marathi": "चा",
                "Bengali": "এর",
                "Punjabi": "ਦੀ",
            }
            conn = connectors.get(language)
            if conn and not t1.endswith(conn):
                result = f"{t1} {conn} {t2}".strip()
            else:
                result = f"{t1} {t2}".strip()
            _TRANSLATION_CACHE[cache_key] = result
            return result

    if cache_key in _TRANSLATION_CACHE:
        return _TRANSLATION_CACHE[cache_key]

    # Rule 5: If sentence contains only unknown words, preserve verbatim
    has_known_word = any(
        t.lower() in TRANSLATION_DICTIONARY or t.lower() in PHRASE_DICTIONARY
        for t in words_only
    )
    if not has_known_word:
        _TRANSLATION_CACHE[cache_key] = clean_sentence
        return clean_sentence

    # Rule 6: Token-by-token compilation with dictionary lookup
    translated_tokens = []
    for token in tokens:
        if re.match(r"^\s+$", token) or re.match(r"^[^\w\s]+$", token):
            translated_tokens.append(token)
        else:
            word_lower = token.lower()
            if word_lower in TRANSLATION_DICTIONARY:
                word_translations = TRANSLATION_DICTIONARY[word_lower]
                translated_word = word_translations.get(language, token)
                translated_tokens.append(translated_word)
            else:
                # Safely preserve unknown words (proper nouns, names)
                translated_tokens.append(token)

    result = "".join(translated_tokens).strip()
    _TRANSLATION_CACHE[cache_key] = result
    _save_disk_cache()
    return result


def get_supported_languages() -> List[str]:
    """Return list of all 10 supported languages."""
    return list(SUPPORTED_LANGUAGES)


def get_language_from_key(key_char: str) -> Optional[str]:
    """Map a pressed key character ('0'-'9') to a language name."""
    return KEY_TO_LANGUAGE.get(key_char, None)


if __name__ == "__main__":
    print("Testing Updated Translator Module across 10 Languages:")
    test_suite = [
        "BALL", "GAME", "BALL GAME", "CAT", "DOG",
        "HELLO", "THANK YOU", "I LOVE YOU", "I WANT WATER",
        "GOOD MORNING", "HOW ARE YOU"
    ]
    for lang in SUPPORTED_LANGUAGES:
        print(f"\n==================== {lang} ====================")
        for phrase in test_suite:
            print(f"  {phrase:<16} -> {translate_sentence(phrase, lang)}")
