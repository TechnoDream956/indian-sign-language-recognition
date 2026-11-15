/**
 * ISL Translator — Frontend Application Logic (Dedicated Alphabet & Fingerspelling)
 *
 * Professional AI/Computer-Vision Workspace Frontend
 *
 * Features:
 *  - Real-time Hand Tracking Skeleton Overlay on HTML5 Canvas (21 keypoints, bone lines)
 *  - Adaptive Paced Video Streaming with Dynamic RTT Latency Monitoring & Monotonic Frame Drops
 *  - Persistent Stable Letter Display & Smooth Latch Progress Tracking
 *  - Multilingual Translation with Real-time & Manual Triggering
 *  - Client-Side Web Speech API (TTS) with Native Indic Voice Routing & Fallbacks
 *  - Play / Stop / Replay Audio Controls
 *  - Manual Sign Commitment (+ Commit Sign)
 *  - Safe Camera Session Toggle (prevents unintended server shutdowns)
 *  - System Settings Modal (Backend URL, Latency Ping Test, Mirroring, Confidence Threshold)
 *  - Keyboard Shortcuts Guide and Full Lifecycle Resilience
 */

"use strict";

// ------------------------------------------------------------------ //
//  Development / Diagnostics Configuration                            //
// ------------------------------------------------------------------ //

const DEV_MOCK_MODE = false;

// ------------------------------------------------------------------ //
//  API & Backend Endpoint Configuration                             //
// ------------------------------------------------------------------ //

const RENDER_BACKEND_URL = "https://isl-recognition-backend.onrender.com";

function isLocalhostEnvironment() {
  return (
    typeof window !== "undefined" &&
    window.location &&
    (window.location.hostname === "localhost" ||
     window.location.hostname === "127.0.0.1" ||
     window.location.protocol === "file:")
  );
}

function getApiUrl(endpoint) {
  let base = window.ISL_BACKEND_URL || localStorage.getItem("isl_backend_url");
  if (!base) {
    base = isLocalhostEnvironment() ? "" : RENDER_BACKEND_URL;
  }
  if (!base) return endpoint;
  const cleanBase = base.replace(/\/+$/, "");
  const cleanEndpoint = endpoint.startsWith("/") ? endpoint : `/${endpoint}`;
  return `${cleanBase}${cleanEndpoint}`;
}

// ------------------------------------------------------------------ //
//  Session Security & Cryptographic Token State                       //
// ------------------------------------------------------------------ //

let feedbackSessionToken = (function () {
  try {
    // Purge legacy persistent localStorage token to prevent long-lived disk leakage
    localStorage.removeItem("isl_feedback_session_token");
    return sessionStorage.getItem("isl_feedback_session_token") || null;
  } catch (e) {
    return null;
  }
})();

function saveFeedbackSessionToken(token) {
  if (!token || typeof token !== "string") return;
  feedbackSessionToken = token.trim();
  try {
    // Store exclusively in sessionStorage for anonymous browser session lifetime
    sessionStorage.setItem("isl_feedback_session_token", feedbackSessionToken);
    localStorage.removeItem("isl_feedback_session_token");
  } catch (e) {}
}

function getFeedbackHeaders(customHeaders = {}) {
  const h = { ...customHeaders };
  if (feedbackSessionToken) {
    h["X-Session-Token"] = feedbackSessionToken;
  }
  return h;
}

function checkResponseSessionToken(resp) {
  try {
    if (!resp || !resp.headers) return;
    const tokenHeader = resp.headers.get("X-Session-Token");
    if (tokenHeader) {
      saveFeedbackSessionToken(tokenHeader);
    }
  } catch (e) {}
}


// ------------------------------------------------------------------ //
//  Constants & Keyboard Map                                          //
// ------------------------------------------------------------------ //

const KEY_TO_LANG = {
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
};

const BCP47_LANG_MAP = {
  "English": "en-IN",
  "Hindi": "hi-IN",
  "Marathi": "mr-IN",
  "Bengali": "bn-IN",
  "Tamil": "ta-IN",
  "Telugu": "te-IN",
  "Gujarati": "gu-IN",
  "Kannada": "kn-IN",
  "Malayalam": "ml-IN",
  "Punjabi": "pa-IN",
};

// MediaPipe 21 Hand Landmarks Skeleton Connectivity
const HAND_CONNECTIONS = [
  // Palm Base
  [0, 1], [0, 5], [5, 9], [9, 13], [13, 17], [0, 17],
  // Thumb
  [1, 2], [2, 3], [3, 4],
  // Index Finger
  [5, 6], [6, 7], [7, 8],
  // Middle Finger
  [9, 10], [10, 11], [11, 12],
  // Ring Finger
  [13, 14], [14, 15], [15, 16],
  // Pinky Finger
  [17, 18], [18, 19], [19, 20]
];

// ------------------------------------------------------------------ //
//  DOM Element References                                            //
// ------------------------------------------------------------------ //

// Camera & Viewport
const $cameraImg            = document.getElementById("camera-img");
const $browserWebcamVideo   = document.getElementById("browser-webcam-video");
const $landmarkCanvas       = document.getElementById("landmark-overlay-canvas");
const $cameraHud            = document.getElementById("camera-hud");
const $btnToggleHud         = document.getElementById("btn-toggle-hud");
const $cameraOverlay        = document.getElementById("camera-overlay");
const $cameraOverlayMsg     = document.getElementById("camera-overlay-msg");
const $btnReconnectStream   = document.getElementById("btn-reconnect-stream");
const $cameraLoading        = document.getElementById("camera-loading");
const $liveIndicator        = document.getElementById("live-indicator");
const $hudModePill          = document.getElementById("hud-mode-pill");
const $fpsBadge             = document.getElementById("fps-badge");
const $hudFpsPill           = document.getElementById("hud-fps-pill");
const $hudLatencyPill       = document.getElementById("hud-latency-pill");
const $hudHandPill          = document.getElementById("hud-hand-pill");
const $hudHandText          = document.getElementById("hud-hand-text");
const $cameraHintText       = document.getElementById("camera-hint-text");

// Multi-Modal Decision Pathways
const $fusionModePill       = document.getElementById("fusion-mode-pill");
const $pathwayLandmarkModel = document.getElementById("pathway-landmark-model");
const $pathwayLandmarkStatus= document.getElementById("pathway-landmark-status");
const $pathwayLandmarkPred  = document.getElementById("pathway-landmark-pred");
const $pathwayLandmarkConf  = document.getElementById("pathway-landmark-conf");
const $pathwayLandmarkTop   = document.getElementById("pathway-landmark-top");
const $pathwayCnnStatus     = document.getElementById("pathway-cnn-status");
const $pathwayCnnPred       = document.getElementById("pathway-cnn-pred");
const $pathwayCnnConf       = document.getElementById("pathway-cnn-conf");
const $pathwayCnnTop        = document.getElementById("pathway-cnn-top");
const $pathwayFusedPred     = document.getElementById("pathway-fused-pred");
const $pathwayFusedConf     = document.getElementById("pathway-fused-conf");
const $pathwayDecisionStatus= document.getElementById("pathway-decision-status");
const $pathwayFusedDecision = document.getElementById("pathway-fused-decision");
const $pathwayFusionReason  = document.getElementById("pathway-fusion-reason");

// System Telemetry Bar
const $statusCamera         = document.getElementById("status-camera");
const $statusCameraVal      = document.getElementById("status-camera-val");
const $statusModel          = document.getElementById("status-model");
const $statusModelVal       = document.getElementById("status-model-val");
const $statusSpeech         = document.getElementById("status-speech");
const $statusSpeechVal      = document.getElementById("status-speech-val");
const $statusSse            = document.getElementById("status-sse");
const $statusSseVal         = document.getElementById("status-sse-val");
const $footerConnectionStat = document.getElementById("footer-connection-stat");

// Hand Tracking Telemetry
const $telemetryHandCount   = document.getElementById("telemetry-hand-count");
const $sensorPrimaryHand    = document.getElementById("sensor-primary-hand");
const $badgePrimaryHand     = document.getElementById("badge-primary-hand");
const $valPrimaryStatus     = document.getElementById("val-primary-status");
const $sensorSecondaryHand  = document.getElementById("sensor-secondary-hand");
const $badgeSecondaryHand   = document.getElementById("badge-secondary-hand");
const $valSecondaryStatus   = document.getElementById("val-secondary-status");
const $valSecondarySafety   = document.getElementById("val-secondary-safety");

// Recognition Hero
const $alphaBigChar         = document.getElementById("alpha-big-char");
const $alphaConfBar         = document.getElementById("alpha-conf-bar");
const $alphaConfNum         = document.getElementById("alpha-conf-num");
const $alphaStableChar      = document.getElementById("alpha-stable-char");
const $alphaEmittedChar     = document.getElementById("alpha-emitted-char");
const $alphaLatchFill       = document.getElementById("alpha-latch-fill");
const $alphaStateBadge      = document.getElementById("alpha-state-badge");
const $btnCommitSign        = document.getElementById("btn-commit-sign");

// WordBuffer & Composer
const $alphaCurrentWord     = document.getElementById("alpha-current-word");
const $alphaSentenceText    = document.getElementById("alpha-sentence-text");
const $alphaWordCount       = document.getElementById("alpha-word-count");
const $btnCopySentence      = document.getElementById("btn-copy-sentence");

// Translation & Audio
const $alphaLangName        = document.getElementById("alpha-lang-name");
const $alphaVoiceBadge      = document.getElementById("alpha-voice-badge");
const $alphaTranslatedText  = document.getElementById("alpha-translated-text");
const $btnAlphaTranslate    = document.getElementById("btn-alpha-translate");
const $btnAlphaSpeakTrans   = document.getElementById("btn-alpha-speak-trans");
const $btnAlphaSpeakOrig    = document.getElementById("btn-alpha-speak-orig");
const $btnAlphaStopSpeech   = document.getElementById("btn-alpha-stop-speech");
const $btnAlphaReplaySpeech = document.getElementById("btn-alpha-replay-speech");
const $langButtons          = document.querySelectorAll(".lang-btn");

// Composition Controls
const $btnAlphaSpace        = document.getElementById("btn-alpha-space");
const $btnAlphaBackspace    = document.getElementById("btn-alpha-backspace");
const $btnAlphaCommit       = document.getElementById("btn-alpha-commit");
const $btnAlphaClearWord    = document.getElementById("btn-alpha-clear-word");
const $btnAlphaClearSent    = document.getElementById("btn-alpha-clear-sent");
const $btnAlphaReset        = document.getElementById("btn-alpha-reset");

// History Log
const $historyCountBadge    = document.getElementById("history-count-badge");
const $historyLogContainer  = document.getElementById("history-log-container");
const $historyEmptyState    = document.getElementById("history-empty-state");
const $historyList          = document.getElementById("history-list");
const $btnClearHistory      = document.getElementById("btn-clear-history");

// Header Actions & Modals
const $btnBackendSettings   = document.getElementById("btn-backend-settings");
const $btnToggleShortcuts   = document.getElementById("btn-toggle-shortcuts");
const $btnSessionToggle     = document.getElementById("btn-session-toggle");
const $sessionBtnText       = document.getElementById("session-btn-text");
const $btnQuitApp           = document.getElementById("btn-quit-app");
const $modalShortcuts       = document.getElementById("modal-shortcuts");
const $btnCloseShortcuts    = document.getElementById("btn-close-shortcuts");
const $btnDismissShortcuts  = document.getElementById("btn-dismiss-shortcuts");
const $modalSettings        = document.getElementById("modal-settings");
const $btnCloseSettings     = document.getElementById("btn-close-settings");
const $btnSaveSettings      = document.getElementById("btn-save-settings");
const $btnResetSettings     = document.getElementById("btn-reset-settings");
const $settingBackendUrl    = document.getElementById("setting-backend-url");
const $btnPingBackend       = document.getElementById("btn-ping-backend");
const $pingStatusRow        = document.getElementById("ping-status-row");
const $pingDot              = document.getElementById("ping-dot");
const $pingText             = document.getElementById("ping-text");
const $settingMirrorToggle  = document.getElementById("setting-mirror-toggle");
const $settingConfThreshold = document.getElementById("setting-conf-threshold");
const $settingConfVal       = document.getElementById("setting-conf-val");
const $settingFpsTarget     = document.getElementById("setting-fps-target");
const $settingFpsVal        = document.getElementById("setting-fps-val");
const $appShutdownOverlay   = document.getElementById("app-shutdown-overlay");
const $toastContainer       = document.getElementById("toast-container");

// ------------------------------------------------------------------ //
//  Local Application State                                            //
// ------------------------------------------------------------------ //

let activeLang = "English";
let sseAbortController = null;
let isSseConnecting = false;
let sseReconnectTimer = null;
let sseReconnectAttempts = 0;
let lastKnownEmittedLetter = null;
let lastProcessedEmissionId = null;
let historyLogEntries = [];
let hudVisible = true;
let voiceCatalog = {};

// Recognition Cache & Diagnostics
let latestRawPrediction = "--";
let latestConfirmedLetter = "--";
let latestConfidence = 0.0;

// Streaming Engine State
let isCloudMode = false;
let browserWebcamStream = null;
let streamingActive = false;
let streamLoopTimer = null;
let browserWebcamCanvas = null;
let monotonicFrameSeq = 0;
let lastCommittedFrameSeq = 0;
let userTargetFps = 12;
let userConfThreshold = 0.60;
let isMirrored = true;

// Speech Synthesis State
let lastSpokenText = "";
let lastSpokenLang = "English";

// ------------------------------------------------------------------ //
//  Toast Notification Utility                                         //
// ------------------------------------------------------------------ //

function showToast(message, type = "info") {
  if (!$toastContainer) return;
  const toast = document.createElement("div");
  toast.className = `toast-item toast-${type}`;
  toast.textContent = message;
  $toastContainer.appendChild(toast);

  setTimeout(() => {
    toast.classList.add("fade-out");
    setTimeout(() => {
      if (toast.parentNode) toast.parentNode.removeChild(toast);
    }, 300);
  }, 2800);
}

// ------------------------------------------------------------------ //
//  Hand Skeleton Canvas Overlay Renderer                              //
// ------------------------------------------------------------------ //

function renderLandmarks(landmarks) {
  if (!$landmarkCanvas) return;

  // Synchronize canvas coordinate resolution with media dimensions
  const videoEl = $browserWebcamVideo || document.getElementById("browser-webcam-video");
  if (videoEl && videoEl.videoWidth && videoEl.videoHeight) {
    if ($landmarkCanvas.width !== videoEl.videoWidth || $landmarkCanvas.height !== videoEl.videoHeight) {
      $landmarkCanvas.width = videoEl.videoWidth;
      $landmarkCanvas.height = videoEl.videoHeight;
    }
  } else if ($cameraImg && $cameraImg.naturalWidth && $cameraImg.naturalHeight) {
    if ($landmarkCanvas.width !== $cameraImg.naturalWidth || $landmarkCanvas.height !== $cameraImg.naturalHeight) {
      $landmarkCanvas.width = $cameraImg.naturalWidth;
      $landmarkCanvas.height = $cameraImg.naturalHeight;
    }
  }

  const ctx = $landmarkCanvas.getContext("2d");
  const width = $landmarkCanvas.width;
  const height = $landmarkCanvas.height;

  ctx.clearRect(0, 0, width, height);

  if (!landmarks || !Array.isArray(landmarks) || landmarks.length === 0) {
    return;
  }

  landmarks.forEach((hand, handIdx) => {
    const points = hand.points || [];
    if (!points || points.length < 21) return;

    // Palette: Hand 0 Cyan/Emerald, Hand 1 Purple/Rose
    const boneColor = handIdx === 0 ? "rgba(6, 182, 212, 0.9)" : "rgba(168, 85, 247, 0.9)";
    const jointColor = handIdx === 0 ? "#10b981" : "#f43f5e";

    // 1. Draw Skeleton Bone Lines (single batched stroke for high FPS)
    ctx.beginPath();
    ctx.strokeStyle = boneColor;
    ctx.lineWidth = 3.0;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    HAND_CONNECTIONS.forEach(([i, j]) => {
      const p1 = points[i];
      const p2 = points[j];
      if (p1 && p2) {
        ctx.moveTo(p1[0] * width, p1[1] * height);
        ctx.lineTo(p2[0] * width, p2[1] * height);
      }
    });
    ctx.stroke();

    // 2. Draw Landmark Joint Nodes
    points.forEach((pt, idx) => {
      const x = pt[0] * width;
      const y = pt[1] * height;
      const isTip = [4, 8, 12, 16, 20].includes(idx);
      const isWrist = idx === 0;

      ctx.beginPath();
      const radius = isTip ? 5.5 : (isWrist ? 6.0 : 3.5);
      ctx.arc(x, y, radius, 0, 2 * Math.PI);
      ctx.fillStyle = isTip ? "#fbbf24" : (isWrist ? "#38bdf8" : jointColor);
      ctx.fill();
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.4;
      ctx.stroke();
    });
  });
}

// ------------------------------------------------------------------ //
//  Client-Side Web Speech API (TTS Engine)                           //
// ------------------------------------------------------------------ //

function getBrowserVoiceForLang(langName) {
  if (!window.speechSynthesis) return null;
  const voices = window.speechSynthesis.getVoices();
  if (!voices || voices.length === 0) return null;

  const targetTag = (BCP47_LANG_MAP[langName] || "en-IN").toLowerCase();
  const shortLang = targetTag.split("-")[0];

  // 1. Exact BCP-47 tag match (e.g., hi-IN, mr-IN)
  let found = voices.find(v => {
    const vLang = v.lang.toLowerCase().replace("_", "-");
    return vLang === targetTag;
  });
  if (found) return found;

  // 2. Primary language prefix match (e.g., hi, mr, bn, ta)
  found = voices.find(v => v.lang.toLowerCase().startsWith(shortLang));
  if (found) return found;

  // 3. Name substring match
  found = voices.find(v => v.name.toLowerCase().includes(langName.toLowerCase()));
  if (found) return found;

  // 4. Default English voice fallback
  return voices.find(v => v.lang.toLowerCase().startsWith("en")) || voices[0];
}

// ------------------------------------------------------------------ //
//  Speech Synthesis State & Multi-Engine Dispatcher                  //
// ------------------------------------------------------------------ //

let isNeuralTTSEnabled = (function () {
  try {
    return localStorage.getItem("isl_neural_tts") === "1";
  } catch (e) {
    return false;
  }
})();
let currentNeuralAudio = null;
let speechRequestSeq = 0;

function speakTextInBrowser(cleanText, langName) {
  if (!("speechSynthesis" in window)) {
    console.warn("[TTS] Browser Web Speech API not supported. Falling back to server trigger.");
    triggerServerSpeech(langName === "English" ? "original" : "translated");
    return;
  }

  window.speechSynthesis.cancel();

  const utterance = new SpeechSynthesisUtterance(cleanText);
  const voice = getBrowserVoiceForLang(langName);
  if (voice) {
    utterance.voice = voice;
    utterance.lang = voice.lang;
  } else {
    utterance.lang = BCP47_LANG_MAP[langName] || "en-IN";
  }

  utterance.rate = 0.95;
  utterance.pitch = 1.0;

  utterance.onstart = () => {
    const btn = langName === "English" ? $btnAlphaSpeakOrig : $btnAlphaSpeakTrans;
    if (btn) btn.classList.add("speaking");
    if ($statusSpeech && $statusSpeechVal) {
      $statusSpeech.className = "telemetry-dot pulse-active";
      $statusSpeechVal.textContent = "SPEAKING";
    }
  };

  utterance.onend = () => {
    const btn = langName === "English" ? $btnAlphaSpeakOrig : $btnAlphaSpeakTrans;
    if (btn) btn.classList.remove("speaking");
    if ($statusSpeech && $statusSpeechVal) {
      $statusSpeech.className = "telemetry-dot ok";
      $statusSpeechVal.textContent = "READY";
    }
  };

  utterance.onerror = (err) => {
    console.warn("[TTS] Web Speech Synthesis error:", err);
    const btn = langName === "English" ? $btnAlphaSpeakOrig : $btnAlphaSpeakTrans;
    if (btn) btn.classList.remove("speaking");
    if ($statusSpeech && $statusSpeechVal) {
      $statusSpeech.className = "telemetry-dot warn";
      $statusSpeechVal.textContent = "IDLE";
    }
  };

  window.speechSynthesis.speak(utterance);
  const preview = cleanText.length > 25 ? cleanText.slice(0, 25) + "..." : cleanText;
  showToast(`Speaking (${langName}): "${preview}"`, "info");
}

function speakTextNeural(cleanText, langName) {
  const reqSeq = ++speechRequestSeq;
  const btn = langName === "English" ? $btnAlphaSpeakOrig : $btnAlphaSpeakTrans;
  if (btn) btn.classList.add("speaking");
  if ($statusSpeech && $statusSpeechVal) {
    $statusSpeech.className = "telemetry-dot pulse-active";
    $statusSpeechVal.textContent = "SYNTHESIZING...";
  }

  fetch(getApiUrl("/api/tts/speak"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: cleanText, language: langName }),
  })
    .then((r) => r.json())
    .then((data) => {
      // Discard stale response if a newer speech request or stop was issued
      if (reqSeq !== speechRequestSeq) {
        return;
      }

      if (data && data.success && data.data_uri) {
        if (currentNeuralAudio) {
          try {
            currentNeuralAudio.pause();
            currentNeuralAudio.src = "";
          } catch (e) {}
          currentNeuralAudio = null;
        }

        const audio = new Audio(data.data_uri);
        currentNeuralAudio = audio;

        audio.onplay = () => {
          if (btn) btn.classList.add("speaking");
          if ($statusSpeech && $statusSpeechVal) {
            $statusSpeech.className = "telemetry-dot pulse-active";
            $statusSpeechVal.textContent = "SPEAKING";
          }
        };

        audio.onended = () => {
          if (btn) btn.classList.remove("speaking");
          if ($statusSpeech && $statusSpeechVal) {
            $statusSpeech.className = "telemetry-dot ok";
            $statusSpeechVal.textContent = "READY";
          }
          if (currentNeuralAudio === audio) {
            currentNeuralAudio = null;
          }
        };

        audio.onerror = (err) => {
          console.warn("[TTS] Neural audio playback failed. Falling back to browser speech:", err);
          if (currentNeuralAudio === audio) {
            currentNeuralAudio = null;
          }
          speakTextInBrowser(cleanText, langName);
        };

        audio.play().catch((playErr) => {
          console.warn("[TTS] Audio play() rejected:", playErr);
          if (currentNeuralAudio === audio) {
            currentNeuralAudio = null;
          }
          speakTextInBrowser(cleanText, langName);
        });

        const preview = cleanText.length > 25 ? cleanText.slice(0, 25) + "..." : cleanText;
        showToast(`Neural Speech (${langName}): "${preview}"`, "info");
      } else {
        // Fallback returned or synthesis failed
        if (btn) btn.classList.remove("speaking");
        speakTextInBrowser(cleanText, langName);
      }
    })
    .catch((netErr) => {
      if (reqSeq !== speechRequestSeq) return;
      console.warn("[TTS] Neural speech request failed. Falling back to browser speech:", netErr);
      if (btn) btn.classList.remove("speaking");
      speakTextInBrowser(cleanText, langName);
    });
}

function speakTextClient(text, langName = "English") {
  if (!text || text === "—" || text.trim() === "") {
    showToast("No sentence text available to speak.", "info");
    return;
  }
  const cleanText = text.trim();
  lastSpokenText = cleanText;
  lastSpokenLang = langName;

  // Stop any ongoing speech playback
  if (currentNeuralAudio) {
    try {
      currentNeuralAudio.pause();
      currentNeuralAudio.src = "";
    } catch (e) {}
    currentNeuralAudio = null;
  }
  if ("speechSynthesis" in window) {
    window.speechSynthesis.cancel();
  }

  if (isNeuralTTSEnabled) {
    speakTextNeural(cleanText, langName);
  } else {
    speakTextInBrowser(cleanText, langName);
  }
}

function stopSpeechClient() {
  speechRequestSeq++;
  if (currentNeuralAudio) {
    try {
      currentNeuralAudio.pause();
      currentNeuralAudio.src = "";
    } catch (e) {}
    currentNeuralAudio = null;
  }
  if ("speechSynthesis" in window) {
    window.speechSynthesis.cancel();
  }
  if ($btnAlphaSpeakTrans) $btnAlphaSpeakTrans.classList.remove("speaking");
  if ($btnAlphaSpeakOrig) $btnAlphaSpeakOrig.classList.remove("speaking");
  if ($statusSpeech && $statusSpeechVal) {
    $statusSpeech.className = "telemetry-dot ok";
    $statusSpeechVal.textContent = "STOPPED";
    setTimeout(() => {
      if ($statusSpeechVal.textContent === "STOPPED") {
        $statusSpeechVal.textContent = "READY";
      }
    }, 1200);
  }
  showToast("Audio playback stopped.", "info");
}

function replaySpeechClient() {
  if (!lastSpokenText) {
    const currentTrans = ($alphaTranslatedText?.textContent || "").trim();
    const currentSent = ($alphaSentenceText?.textContent || "").trim();
    if (currentTrans && currentTrans !== "—") {
      speakTextClient(currentTrans, activeLang);
    } else if (currentSent && currentSent !== "—") {
      speakTextClient(currentSent, "English");
    } else {
      showToast("No speech audio to replay.", "info");
    }
    return;
  }
  speakTextClient(lastSpokenText, lastSpokenLang);
}

// ------------------------------------------------------------------ //
//  TTS Voice Catalog Discovery                                        //
// ------------------------------------------------------------------ //

async function loadVoiceCatalog() {
  try {
    const res = await fetch(getApiUrl("/api/tts/voices"));
    if (res.ok) {
      const data = await res.json();
      voiceCatalog = data.catalog || {};
      updateVoiceBadgeUI(activeLang);
    }
  } catch (err) {
    console.warn("[TTS] Could not load server voice catalog:", err);
    updateVoiceBadgeUI(activeLang);
  }
}

function updateVoiceBadgeUI(lang) {
  if (!$alphaVoiceBadge) return;
  const browserVoice = getBrowserVoiceForLang(lang);
  if (browserVoice) {
    $alphaVoiceBadge.textContent = `Voice: ${browserVoice.name} (${lang})`;
    $alphaVoiceBadge.className = "trans-voice-badge ok";
    return;
  }

  const info = voiceCatalog[lang];
  if (info && info.available && info.voice) {
    $alphaVoiceBadge.textContent = `Voice: ${info.voice} (${lang})`;
    $alphaVoiceBadge.className = "trans-voice-badge ok";
  } else if (lang === "English") {
    $alphaVoiceBadge.textContent = "Voice: English (System Default)";
    $alphaVoiceBadge.className = "trans-voice-badge ok";
  } else {
    $alphaVoiceBadge.textContent = `Voice: Client Web TTS (${lang})`;
    $alphaVoiceBadge.className = "trans-voice-badge ok";
  }
}

// ------------------------------------------------------------------ //
//  SSE — Real-Time State Listener (Authenticated Fetch Stream)        //
// ------------------------------------------------------------------ //

function disconnectSSE() {
  clearTimeout(sseReconnectTimer);
  sseReconnectTimer = null;
  isSseConnecting = false;
  if (sseAbortController) {
    try {
      sseAbortController.abort();
    } catch (e) {}
    sseAbortController = null;
  }
  updateSseStatus(false, "PAUSED");
}

function scheduleSseReconnect(delayMs) {
  if (isSessionTerminated) return;
  clearTimeout(sseReconnectTimer);
  sseReconnectAttempts += 1;
  const backoff = delayMs || Math.min(1000 * Math.pow(1.5, Math.min(sseReconnectAttempts, 6)), 15000);
  sseReconnectTimer = setTimeout(() => {
    connectSSE();
  }, backoff);
}

async function connectSSE() {
  if (DEV_MOCK_MODE) {
    runDevMockLoop();
    return;
  }
  if (isSessionTerminated) {
    return;
  }

  // Cancel any existing stream and prevent duplicate connections
  if (sseAbortController) {
    try {
      sseAbortController.abort();
    } catch (e) {}
    sseAbortController = null;
  }

  if (isSseConnecting) {
    return;
  }

  isSseConnecting = true;
  updateSseStatus(false, "CONNECTING");

  // Ensure session token exists before establishing authenticated stream
  if (!feedbackSessionToken) {
    try {
      await initSessionToken();
    } catch (e) {}
  }

  const controller = new AbortController();
  sseAbortController = controller;

  try {
    const headers = getFeedbackHeaders({
      "Accept": "text/event-stream",
      "Cache-Control": "no-cache",
    });

    // Connect strictly without query-string tokens (X-Session-Token header only)
    const response = await fetch(getApiUrl("/events"), {
      method: "GET",
      headers: headers,
      signal: controller.signal,
    });

    checkResponseSessionToken(response);

    if (response.status === 401 || response.status === 403) {
      console.warn(`[SSE] Authentication error (${response.status}). Refreshing session...`);
      updateSseStatus(false, response.status === 403 ? "FORBIDDEN" : "UNAUTHORIZED");
      isSseConnecting = false;
      feedbackSessionToken = null;
      try {
        sessionStorage.removeItem("isl_feedback_session_token");
      } catch (e) {}
      await initSessionToken();
      scheduleSseReconnect(3500);
      return;
    }

    if (!response.ok || !response.body) {
      throw new Error(`SSE HTTP error ${response.status}: ${response.statusText}`);
    }

    // Successfully connected!
    isSseConnecting = false;
    sseReconnectAttempts = 0;
    updateSseStatus(true, "LIVE");
    if ($cameraOverlay && $cameraOverlay.dataset.source === "sse") {
      $cameraOverlay.style.display = "none";
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) {
        break;
      }
      buffer += decoder.decode(value, { stream: true });

      const parts = buffer.split(/\r?\n\r?\n/);
      buffer = parts.pop() || "";

      for (const part of parts) {
        const trimmed = part.trim();
        if (!trimmed) continue;

        const lines = trimmed.split(/\r?\n/);
        for (const line of lines) {
          if (line.startsWith("data:")) {
            const rawJson = line.slice(5).trim();
            if (rawJson) {
              try {
                const data = JSON.parse(rawJson);
                if (data.landmarks) {
                  renderLandmarks(data.landmarks);
                }
                applyState(data);
              } catch (e) {
                console.warn("[SSE] Parse error on payload:", e);
              }
            }
          }
        }
      }
    }

    if (!controller.signal.aborted && !isSessionTerminated) {
      scheduleSseReconnect(1500);
    }
  } catch (err) {
    isSseConnecting = false;
    if (controller.signal.aborted || isSessionTerminated) {
      return;
    }
    console.warn("[SSE] Stream interrupted:", err.message || err);
    updateSseStatus(false, "RECONNECTING");
    scheduleSseReconnect();
  }
}

function updateSseStatus(isLive, labelText) {
  if ($statusSse && $statusSseVal) {
    $statusSse.className = `telemetry-dot ${isLive ? "ok" : "warn"}`;
    $statusSseVal.textContent = labelText;
  }
  if ($footerConnectionStat) {
    $footerConnectionStat.textContent = `SSE: ${labelText}`;
  }
}

// ------------------------------------------------------------------ //
//  Paced Adaptive Streaming Engine                                   //
// ------------------------------------------------------------------ //

async function enableCloudWebcamMode() {
  if (isCloudMode && browserWebcamStream && streamingActive) return;
  isCloudMode = true;

  if ($cameraHintText) {
    $cameraHintText.textContent = "Cloud Mode: Adaptive browser webcam ingestion with real-time skeleton overlay";
  }
  if ($hudModePill) {
    $hudModePill.style.display = "inline-flex";
    $hudModePill.textContent = "CLOUD WEBCAM";
  }

  const videoEl = $browserWebcamVideo || document.getElementById("browser-webcam-video");
  if (!videoEl) {
    console.error("[Camera] browser-webcam-video element missing in DOM");
    return;
  }

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showToast("Browser webcam API not supported on this browser.", "error");
    if ($cameraOverlay && $cameraOverlayMsg) {
      $cameraOverlay.style.display = "flex";
      $cameraOverlayMsg.textContent = "Browser webcam API not supported on this device/browser.";
    }
    return;
  }

  let stream = null;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        facingMode: "user"
      },
      audio: false
    });
  } catch (err) {
    console.warn("[Camera] getUserMedia failed:", err);
    if ($cameraOverlay) {
      $cameraOverlay.style.display = "flex";
      if ($cameraOverlayMsg) {
        $cameraOverlayMsg.textContent = "Webcam permission required. Click Reconnect to grant access.";
      }
    }
    const isPermissionError = (
      err.name === "NotAllowedError" ||
      err.name === "PermissionDeniedError" ||
      err.name === "SecurityError"
    );
    const msg = isPermissionError
      ? "Webcam permission denied: " + err.message
      : "Webcam device error: " + (err.message || err.name);
    showToast(msg, "error");
    return;
  }

  try {
    browserWebcamStream = stream;
    videoEl.srcObject = stream;
    await videoEl.play();
    videoEl.style.display = "block";

    if ($landmarkCanvas) {
      $landmarkCanvas.style.display = "block";
      syncMirroring();
    }
    if ($cameraImg) {
      $cameraImg.style.display = "none";
    }
    if ($cameraLoading) {
      $cameraLoading.style.display = "none";
    }
    if ($cameraOverlay) {
      $cameraOverlay.style.display = "none";
    }
  } catch (err) {
    console.error("[Camera] Video initialization error:", err);
    showToast("Video display error: " + err.message, "error");
    return;
  }

  try {
    startCameraSession();
    showToast("Webcam connected. Vision tracking active.", "success");
  } catch (err) {
    console.error("[Camera] Session activation error:", err);
    showToast("Camera session activation error: " + err.message, "error");
  }
}

// ------------------------------------------------------------------ //
//  Camera Lifecycle State Machine Subsystem                          //
// ------------------------------------------------------------------ //

let cameraLifecycleState = "STOPPED"; // "STOPPED" | "STARTING" | "RUNNING" | "STOPPING"

async function startCameraSession() {
  if (isSessionTerminated) return;
  if (cameraLifecycleState === "STARTING" || cameraLifecycleState === "RUNNING") return;

  cameraLifecycleState = "STARTING";
  if ($btnSessionToggle) {
    $btnSessionToggle.classList.add("transitioning");
    $btnSessionToggle.disabled = true;
  }

  if (streamLoopTimer) {
    clearTimeout(streamLoopTimer);
    streamLoopTimer = null;
  }

  const isLocal = isLocalhostEnvironment();
  const videoEl = $browserWebcamVideo || document.getElementById("browser-webcam-video");

  streamingActive = true;

  try {
    if (!isCloudMode && isLocal && $cameraImg) {
      // Local hardware camera (Flask MJPEG stream)
      $cameraImg.src = getApiUrl("/video_feed?t=" + Date.now());
      $cameraImg.style.display = "block";
      if (videoEl) videoEl.style.display = "none";
      if ($landmarkCanvas) {
        $landmarkCanvas.style.display = "block";
      }
      fetch(getApiUrl("/api/camera/start"), { method: "POST" }).catch(() => {});
    } else if (browserWebcamStream && videoEl) {
      // Cloud browser webcam: stream is active, resume playback & start streaming loop
      await videoEl.play().catch(() => {});
      videoEl.style.display = "block";
      if ($cameraImg) $cameraImg.style.display = "none";
      if ($cameraOverlay) $cameraOverlay.style.display = "none";
      if ($landmarkCanvas) $landmarkCanvas.style.display = "block";
      streamLoop(videoEl);
    } else {
      // Cloud browser webcam: stream not yet initialized, acquire media
      streamingActive = false;
      cameraLifecycleState = "STOPPED";
      if ($btnSessionToggle) {
        $btnSessionToggle.classList.remove("transitioning");
        $btnSessionToggle.disabled = false;
      }
      enableCloudWebcamMode();
      return;
    }

    cameraLifecycleState = "RUNNING";
    if ($sessionBtnText) $sessionBtnText.textContent = "STOP CAMERA";
    const $sessionIcon = document.getElementById("session-btn-icon");
    if ($sessionIcon) $sessionIcon.textContent = "⏹";

    if ($btnSessionToggle) {
      $btnSessionToggle.classList.remove("stopped", "transitioning");
      $btnSessionToggle.classList.add("running");
      $btnSessionToggle.disabled = false;
      $btnSessionToggle.title = "Stop Camera stream and freeze recognition";
    }

    const $liveIndicator = document.getElementById("live-indicator");
    if ($liveIndicator) {
      $liveIndicator.textContent = "LIVE";
      $liveIndicator.className = "stream-indicator live";
    }

    // Resume authenticated SSE stream
    if (!isSessionTerminated) {
      connectSSE();
    }

    showToast("Camera started. Recognition active.", "success");
  } catch (err) {
    console.error("[Camera] Start error:", err);
    cameraLifecycleState = "STOPPED";
    streamingActive = false;
    if ($btnSessionToggle) {
      $btnSessionToggle.classList.remove("transitioning");
      $btnSessionToggle.disabled = false;
    }
    showToast("Camera start error: " + (err.message || err), "error");
  }
}

async function stopCameraSession() {
  if (cameraLifecycleState === "STOPPING" || cameraLifecycleState === "STOPPED") return;

  cameraLifecycleState = "STOPPING";
  if ($btnSessionToggle) {
    $btnSessionToggle.classList.add("transitioning");
    $btnSessionToggle.disabled = true;
  }

  streamingActive = false;
  if (streamLoopTimer) {
    clearTimeout(streamLoopTimer);
    streamLoopTimer = null;
  }

  // Notify backend hardware loop
  fetch(getApiUrl("/api/camera/stop"), { method: "POST" }).catch(() => {});

  // Invalidate in-flight frame responses so pending predictions are discarded
  monotonicFrameSeq += 1000;
  lastCommittedFrameSeq = monotonicFrameSeq;

  // Disconnect SSE stream and cancel background reader
  disconnectSSE();

  // Stop and release all active webcam tracks
  if (browserWebcamStream) {
    browserWebcamStream.getTracks().forEach((track) => {
      try { track.stop(); } catch (e) {}
    });
    browserWebcamStream = null;
  }

  const videoEl = $browserWebcamVideo || document.getElementById("browser-webcam-video");
  if (videoEl) {
    videoEl.pause();
    videoEl.srcObject = null;
  }

  // In local hardware camera mode, stop image refresh
  if ($cameraImg) {
    $cameraImg.dataset.activeSrc = $cameraImg.src;
    $cameraImg.style.display = "none";
  }

  if ($landmarkCanvas) {
    const ctx = $landmarkCanvas.getContext("2d");
    ctx.clearRect(0, 0, $landmarkCanvas.width, $landmarkCanvas.height);
  }

  cameraLifecycleState = "STOPPED";
  if ($sessionBtnText) $sessionBtnText.textContent = "START CAMERA";
  const $sessionIcon = document.getElementById("session-btn-icon");
  if ($sessionIcon) $sessionIcon.textContent = "▶";

  if ($btnSessionToggle) {
    $btnSessionToggle.classList.remove("running", "transitioning");
    $btnSessionToggle.classList.add("stopped");
    $btnSessionToggle.disabled = false;
    $btnSessionToggle.title = "Start Camera stream and resume recognition";
  }

  const $liveIndicator = document.getElementById("live-indicator");
  if ($liveIndicator) {
    $liveIndicator.textContent = "STOPPED";
    $liveIndicator.className = "stream-indicator paused";
  }

  showToast("Camera stopped. Video stream & sensor resources released.", "info");
}

function toggleCameraSession() {
  if (cameraLifecycleState === "STARTING" || cameraLifecycleState === "STOPPING") {
    console.debug("[Camera State] Toggle ignored during transition:", cameraLifecycleState);
    return;
  }
  if (cameraLifecycleState === "RUNNING" || streamingActive) {
    stopCameraSession();
  } else {
    startCameraSession();
  }
}

async function streamLoop(videoEl) {
  if (!streamingActive || !videoEl || videoEl.paused || videoEl.ended || !videoEl.videoWidth) {
    if (streamingActive) {
      streamLoopTimer = setTimeout(() => streamLoop(videoEl), 100);
    }
    return;
  }

  if (!browserWebcamCanvas) {
    browserWebcamCanvas = document.createElement("canvas");
    browserWebcamCanvas.width = 640;
    browserWebcamCanvas.height = 480;
  }
  const ctx = browserWebcamCanvas.getContext("2d");
  ctx.drawImage(videoEl, 0, 0, 640, 480);

  const frameSeq = ++monotonicFrameSeq;
  const startTime = performance.now();

  browserWebcamCanvas.toBlob(async (blob) => {
    if (!blob || !streamingActive) {
      scheduleNextFrame(videoEl, startTime);
      return;
    }

    try {
      const res = await fetch(getApiUrl("/api/camera/frame"), {
        method: "POST",
        headers: getFeedbackHeaders({ "Content-Type": "image/jpeg" }),
        body: blob,
      });
      checkResponseSessionToken(res);

      const rtt = Math.round(performance.now() - startTime);
      if ($hudLatencyPill) {
        $hudLatencyPill.textContent = `RTT: ${rtt}ms`;
        if (rtt < 180) {
          $hudLatencyPill.style.color = "#4ade80";
        } else if (rtt < 380) {
          $hudLatencyPill.style.color = "#fbbf24";
        } else {
          $hudLatencyPill.style.color = "#f87171";
        }
      }

      if (res.ok) {
        const data = await res.json();
        // Latest-Frame-Wins: ignore stale response if newer frame already committed
        if (frameSeq >= lastCommittedFrameSeq) {
          lastCommittedFrameSeq = frameSeq;
          if (data.landmarks) {
            renderLandmarks(data.landmarks);
          }
          if (data.state) {
            applyState(data.state);
          }
          if (data.prediction_id) {
            latestPredictionId = data.prediction_id;
            currentFeedbackPredictionId = data.prediction_id;
            handleFeedbackOpportunity(data);
          }
        }
      }
    } catch (err) {
      console.debug("[Streaming Loop Net Warning]", err);
    } finally {
      scheduleNextFrame(videoEl, startTime);
    }
  }, "image/jpeg", 0.65);
}

function scheduleNextFrame(videoEl, startTime) {
  if (!streamingActive) return;
  const targetInterval = 1000 / userTargetFps;
  const elapsed = performance.now() - startTime;
  const delay = Math.max(5, targetInterval - elapsed);
  streamLoopTimer = setTimeout(() => streamLoop(videoEl), delay);
}

async function checkEnvironmentAndInitWebcam() {
  const urlParams = new URLSearchParams(window.location.search);
  const isRemote = !isLocalhostEnvironment();
  const forceCloud = urlParams.get("mode") === "cloud" || window.FORCE_CLOUD_CAMERA === true || isRemote;

  if (forceCloud) {
    enableCloudWebcamMode();
    return;
  }

  try {
    const res = await fetch(getApiUrl("/api/status"));
    if (res.ok) {
      const data = await res.json();
      if (data.cloud_mode || forceCloud) {
        enableCloudWebcamMode();
      }
    }
  } catch (err) {
    try {
      const hRes = await fetch(getApiUrl("/api/healthz"));
      if (hRes.ok) {
        const hData = await hRes.json();
        if (hData.cloud_mode || forceCloud) {
          enableCloudWebcamMode();
        }
      }
    } catch (e) {}
    if (forceCloud) {
      enableCloudWebcamMode();
    }
  }
}

// ------------------------------------------------------------------ //
//  State Application Engine                                            //
// ------------------------------------------------------------------ //

function applyState(state) {
  if (!state) return;

  // 1. Sync FPS and Stream Indicator
  const fpsVal = state.fps !== undefined ? state.fps : "--";
  if ($fpsBadge) $fpsBadge.textContent = `FPS: ${fpsVal}`;
  if ($hudFpsPill) $hudFpsPill.textContent = `FPS: ${fpsVal}`;

  // 2. Camera Status & Cloud Mode Handling
  const isCloud = state.cloud_mode === true || state.camera_mode === "cloud_browser" || isCloudMode;
  const camOk = state.camera_ok !== false;

  if (isCloud && !isCloudMode) {
    enableCloudWebcamMode();
  }

  if ($statusCamera && $statusCameraVal) {
    $statusCamera.className = `telemetry-dot ${camOk ? "ok" : "err"}`;
    $statusCameraVal.textContent = camOk ? (isCloud ? "CLOUD (BROWSER)" : "READY (LOCAL)") : "OFFLINE";
  }
  if ($liveIndicator) {
    $liveIndicator.className = `stream-indicator ${camOk ? "live" : "offline"}`;
    $liveIndicator.textContent = camOk ? (isCloud ? "CLOUD LIVE" : "LIVE") : "OFFLINE";
  }
  if ($hudModePill) {
    $hudModePill.style.display = isCloud ? "inline-flex" : "none";
    $hudModePill.textContent = "CLOUD WEBCAM";
  }

  // Viewport Overlay on Camera Error
  if (!camOk && !isCloud) {
    if ($cameraOverlay) {
      $cameraOverlay.dataset.source = "camera";
      $cameraOverlay.style.display = "flex";
      if ($cameraOverlayMsg) {
        $cameraOverlayMsg.textContent = state.speech_status && state.speech_status.includes("Camera")
          ? state.speech_status
          : "Webcam feed unavailable. Verify camera connection.";
      }
    }
  } else if ($cameraOverlay && $cameraOverlay.dataset.source === "camera" && camOk) {
    $cameraOverlay.style.display = "none";
  }

  // 3. AI Model Inference Status
  const modelOk = state.model_ok !== false;
  if ($statusModel && $statusModelVal) {
    $statusModel.className = `telemetry-dot ${modelOk ? "ok" : "err"}`;
    $statusModelVal.textContent = modelOk ? "READY" : "ERROR";
  }

  // 4. Speech Synthesis Telemetry
  const speechStatus = state.speech_status || "Ready";
  const isSpeaking = speechStatus === "Speaking..." || (window.speechSynthesis && window.speechSynthesis.speaking);
  if ($statusSpeech && $statusSpeechVal) {
    $statusSpeech.className = `telemetry-dot ${isSpeaking ? "pulse-active" : (speechStatus.includes("error") ? "err" : "ok")}`;
    $statusSpeechVal.textContent = isSpeaking ? "SPEAKING" : speechStatus.toUpperCase();
  }

  // 5. Hand Tracking Telemetry (0, 1, 2 Hands)
  const isHand = Boolean(state.hand_detected || state.alphabet_hand_detected);
  const handCount = isHand ? (state.alphabet_hand_count !== undefined ? state.alphabet_hand_count : 1) : 0;

  if ($hudHandText) {
    if (handCount === 0) {
      $hudHandText.textContent = "NO HAND";
    } else if (handCount === 1) {
      $hudHandText.textContent = "1 HAND (ACTIVE)";
    } else {
      $hudHandText.textContent = "2 HANDS (SAFETY HOLD)";
    }
  }

  if ($hudHandPill) {
    $hudHandPill.classList.remove("no-hand", "one-hand", "dual-hand", "detected");
    if (handCount === 0) {
      $hudHandPill.classList.add("no-hand");
    } else if (handCount === 1) {
      $hudHandPill.classList.add("one-hand", "detected");
    } else {
      $hudHandPill.classList.add("dual-hand", "detected");
    }
  }

  if ($telemetryHandCount) {
    $telemetryHandCount.textContent = `${handCount} ${handCount === 1 ? "Hand" : "Hands"}`;
    $telemetryHandCount.className = `hand-count-pill count-${handCount}`;
  }

  // Detailed Hand Sensors Grid
  if ($badgePrimaryHand && $valPrimaryStatus) {
    if (handCount >= 1) {
      $badgePrimaryHand.textContent = "DETECTED";
      $badgePrimaryHand.className = "sensor-badge detected";
      $valPrimaryStatus.textContent = "Tracking 21 Keypoints";
      if ($sensorPrimaryHand) $sensorPrimaryHand.classList.add("active");
    } else {
      $badgePrimaryHand.textContent = "MISSING";
      $badgePrimaryHand.className = "sensor-badge missing";
      $valPrimaryStatus.textContent = "No Landmark Lock";
      if ($sensorPrimaryHand) $sensorPrimaryHand.classList.remove("active");
    }
  }

  if ($badgeSecondaryHand && $valSecondaryStatus && $valSecondarySafety) {
    if (handCount >= 2) {
      $badgeSecondaryHand.textContent = "DETECTED";
      $badgeSecondaryHand.className = "sensor-badge detected";
      $valSecondaryStatus.textContent = "Tracking 21 Keypoints";
      $valSecondarySafety.textContent = "Dual-Hand Detected (Standby Safety)";
      $valSecondarySafety.className = "detail-value text-amber";
      if ($sensorSecondaryHand) $sensorSecondaryHand.classList.add("active");
    } else {
      $badgeSecondaryHand.textContent = "MISSING";
      $badgeSecondaryHand.className = "sensor-badge missing";
      $valSecondaryStatus.textContent = "Standby";
      $valSecondarySafety.textContent = "Active (Single-Hand Invariant)";
      $valSecondarySafety.className = "detail-value text-muted";
      if ($sensorSecondaryHand) $sensorSecondaryHand.classList.remove("active");
    }
  }

  // 6. Active Target Language
  if (state.language && state.language !== activeLang) {
    activeLang = state.language;
    if ($alphaLangName) $alphaLangName.textContent = activeLang;
    setActiveLanguageUI(activeLang);
    updateVoiceBadgeUI(activeLang);
  }

  // 6b. Multi-Modal Decision Pathways Telemetry & Live Debugger
  if ($fusionModePill && state.fusion_mode) {
    $fusionModePill.textContent = state.fusion_mode;
  }
  if ($pathwayLandmarkModel && (state.active_pipeline || state.landmark_model)) {
    $pathwayLandmarkModel.textContent = state.active_pipeline || state.landmark_model;
  }
  if ($pathwayLandmarkPred) {
    $pathwayLandmarkPred.textContent = state.landmark_prediction || "--";
  }
  if ($pathwayLandmarkConf) {
    const lmConf = parseFloat(state.landmark_confidence || 0);
    $pathwayLandmarkConf.textContent = lmConf > 0 ? `${Math.round(lmConf * 100)}%` : "--";
  }
  if ($pathwayLandmarkTop) {
    if (state.landmark_top5 && state.landmark_top5.length > 0) {
      const topStr = state.landmark_top5.slice(0, 3).map(c => `${c.class} (${Math.round(c.confidence * 100)}%)`).join(", ");
      $pathwayLandmarkTop.textContent = `Top: ${topStr}`;
    } else {
      $pathwayLandmarkTop.textContent = `Top: ${state.landmark_prediction ? state.landmark_prediction : "--"}`;
    }
  }

  if ($pathwayCnnStatus && state.cnn_status) {
    $pathwayCnnStatus.textContent = state.cnn_status;
    $pathwayCnnStatus.className = `pathway-status ${state.cnn_status === "ONLINE" ? "ok" : "offline"}`;
  }
  if ($pathwayCnnPred) {
    $pathwayCnnPred.textContent = state.cnn_prediction || "--";
  }
  if ($pathwayCnnConf) {
    const cnnConf = parseFloat(state.cnn_confidence || 0);
    $pathwayCnnConf.textContent = cnnConf > 0 ? `${Math.round(cnnConf * 100)}%` : "--";
  }
  if ($pathwayCnnTop) {
    if (state.cnn_top5 && state.cnn_top5.length > 0) {
      const topStr = state.cnn_top5.slice(0, 3).map(c => `${c.class} (${Math.round(c.confidence * 100)}%)`).join(", ");
      $pathwayCnnTop.textContent = `Top: ${topStr}`;
    } else {
      $pathwayCnnTop.textContent = `Top: ${state.cnn_prediction ? state.cnn_prediction : "--"}`;
    }
  }

  // Fused Output Diagnostics
  if ($pathwayFusedPred) {
    $pathwayFusedPred.textContent = state.fused_prediction || state.prediction || "--";
  }
  if ($pathwayFusedConf) {
    const fConf = parseFloat(state.fused_confidence !== undefined ? state.fused_confidence : (state.confidence || 0));
    $pathwayFusedConf.textContent = fConf > 0 ? `${Math.round(fConf * 100)}%` : "--";
  }
  if ($pathwayDecisionStatus) {
    const isAccepted = state.is_accepted !== undefined ? state.is_accepted : Boolean(state.fused_prediction || state.prediction);
    $pathwayDecisionStatus.textContent = isAccepted ? "ACCEPTED" : "HOLDING / REJECTED";
    $pathwayDecisionStatus.style.color = isAccepted ? "#4ade80" : "#fbbf24";
  }
  if ($pathwayFusedDecision) {
    $pathwayFusedDecision.textContent = `Mode: ${state.fusion_mode || "STANDBY"}`;
  }
  if ($pathwayFusionReason && state.fusion_reason) {
    $pathwayFusionReason.textContent = state.fusion_reason;
  }

  // 7. Real-Time Gesture Recognition Display
  if (state.prediction_id && !isCloudMode) {
    latestPredictionId = state.prediction_id;
    currentFeedbackPredictionId = state.prediction_id;
  }
  const rawLetter = state.fused_prediction || state.raw_prediction || state.prediction || state.alphabet_raw_letter || state.pred_label || "--";
  const isValidLetter = rawLetter !== "--" && rawLetter !== "...";
  if (isValidLetter) {
    latestRawPrediction = rawLetter;
    latestPredictedLetter = rawLetter;
  }

  if ($alphaBigChar) {
    $alphaBigChar.textContent = isValidLetter ? rawLetter : "--";
    $alphaBigChar.classList.toggle("detected", isValidLetter);
  }

  // Confidence Bar & Percentage
  const alphaConf = parseFloat(state.fused_confidence !== undefined ? state.fused_confidence : (state.alphabet_confidence !== undefined ? state.alphabet_confidence : (state.confidence || 0.0)));
  latestConfidence = alphaConf;
  const alphaPct = Math.min(100, Math.max(0, Math.round(alphaConf * 100)));

  if ($alphaConfNum) {
    $alphaConfNum.textContent = isValidLetter ? `${alphaPct}%` : "0%";
  }
  if ($alphaConfBar) {
    $alphaConfBar.style.width = isValidLetter ? `${alphaPct}%` : "0%";
    if (alphaConf >= 0.70) {
      $alphaConfBar.style.background = "var(--color-primary-accent)";
    } else if (alphaConf >= 0.45) {
      $alphaConfBar.style.background = "var(--color-warm-accent)";
    } else {
      $alphaConfBar.style.background = "var(--color-alert)";
    }
  }

  // Stable Candidate & Debounce Telemetry (Persistent Display)
  const stableCandidate = state.stable_letter || state.alphabet_stable_letter || state.last_stable_letter;
  if (stableCandidate && stableCandidate !== "--") {
    latestConfirmedLetter = stableCandidate;
  }
  if ($alphaStableChar) {
    $alphaStableChar.textContent = latestConfirmedLetter !== "--" ? latestConfirmedLetter : (stableCandidate || "--");
  }

  // Emitted Character & Session History Recording
  const currentEmitted = state.emitted_letter || state.alphabet_emitted_letter;
  const emissionId = (state.alphabet_emission_id !== undefined && state.alphabet_emission_id !== null)
    ? state.alphabet_emission_id
    : (state.emission_id !== undefined ? state.emission_id : null);

  if ($alphaEmittedChar) {
    if (currentEmitted) {
      $alphaEmittedChar.textContent = currentEmitted;
    } else {
      $alphaEmittedChar.textContent = "--";
    }
  }

  if (lastProcessedEmissionId === null && emissionId && !currentEmitted) {
    lastProcessedEmissionId = emissionId;
  }

  if (currentEmitted && emissionId && (lastProcessedEmissionId === null || emissionId > lastProcessedEmissionId)) {
    lastProcessedEmissionId = emissionId;
    if ($alphaEmittedChar) {
      $alphaEmittedChar.classList.remove("pulse");
      void $alphaEmittedChar.offsetWidth;
      $alphaEmittedChar.classList.add("pulse");
    }
    recordHistoryEntry({
      type: "EMITTED",
      glyph: currentEmitted,
      score: alphaPct,
      timestamp: formatTimestamp(new Date()),
    });
  }

  // Engine State Badge & Hold-to-Emit Progress Bar
  const alphaState = (state.alphabet_state || "IDLE").toUpperCase();
  if ($alphaStateBadge) {
    $alphaStateBadge.textContent = alphaState;
    let badgeCategory = "idle";
    if (alphaState.startsWith("STABILIZING") || alphaState === "COLLECTING") {
      badgeCategory = "collecting";
    } else if (alphaState.startsWith("HOLDING")) {
      badgeCategory = "holding";
    } else if (alphaState.startsWith("EMITTED") || alphaState.startsWith("LATCHED")) {
      badgeCategory = "emitted";
    } else if (alphaState === "IDLE" || alphaState === "RESET" || alphaState.startsWith("COOLDOWN")) {
      badgeCategory = "idle";
    }
    $alphaStateBadge.className = `state-badge state-${badgeCategory}`;
  }

  if ($alphaLatchFill) {
    if (state.stability_progress !== undefined && state.stability_progress !== null) {
      const stabPct = Math.min(100, Math.max(0, Math.round(state.stability_progress * 100)));
      $alphaLatchFill.style.width = `${stabPct}%`;
      $alphaLatchFill.style.background = stabPct >= 100 ? "var(--color-primary-accent)" : "var(--color-warm-accent)";
    } else if (alphaState.startsWith("STABILIZING")) {
      const match = alphaState.match(/STABILIZING\((\d+)\/(\d+)\)/);
      if (match) {
        const current = parseInt(match[1], 10);
        const total = parseInt(match[2], 10) || 5;
        const pct = Math.min(100, Math.max(0, Math.round((current / total) * 100)));
        $alphaLatchFill.style.width = `${pct}%`;
      } else {
        $alphaLatchFill.style.width = "50%";
      }
      $alphaLatchFill.style.background = "var(--color-warm-accent)";
    } else if (alphaState.startsWith("HOLDING") || alphaState === "COLLECTING") {
      $alphaLatchFill.style.width = "100%";
      $alphaLatchFill.style.background = "var(--color-warm-accent)";
    } else if (alphaState.startsWith("EMITTED") || alphaState.startsWith("LATCHED")) {
      $alphaLatchFill.style.width = "100%";
      $alphaLatchFill.style.background = "var(--color-primary-accent)";
    } else {
      $alphaLatchFill.style.width = "0%";
    }
  }

  // 8. WordBuffer Assembly Display
  const currentWord = state.alphabet_current_word || "";
  if ($alphaCurrentWord) {
    if (currentWord.trim()) {
      $alphaCurrentWord.textContent = currentWord;
      $alphaCurrentWord.classList.remove("empty");
    } else {
      $alphaCurrentWord.innerHTML = '<span class="word-placeholder">Awaiting fingerspelled letters...</span>';
      $alphaCurrentWord.classList.add("empty");
    }
  }

  const sentence = state.alphabet_sentence || "";
  const hasSentence = Boolean(sentence.trim());
  if ($alphaSentenceText) {
    $alphaSentenceText.textContent = hasSentence ? sentence : "—";
    $alphaSentenceText.classList.toggle("empty", !hasSentence);
  }

  const totalWords = (sentence.trim() ? sentence.trim().split(/\s+/).length : 0) + (currentWord.trim() ? 1 : 0);
  if ($alphaWordCount) {
    $alphaWordCount.textContent = `${totalWords} ${totalWords === 1 ? "Word" : "Words"}`;
  }

  // 9. Multilingual Translation Display
  const translation = state.alphabet_translation || state.translation || "";
  const hasTranslation = Boolean(translation.trim());
  if ($alphaTranslatedText) {
    $alphaTranslatedText.textContent = hasTranslation ? translation : "—";
    $alphaTranslatedText.classList.toggle("empty", !hasTranslation);
  }
}

// ------------------------------------------------------------------ //
//  Recognition Event History Management                              //
// ------------------------------------------------------------------ //

function recordHistoryEntry(entry) {
  historyLogEntries.unshift(entry);
  if (historyLogEntries.length > 25) {
    historyLogEntries.pop();
  }
  renderHistoryLog();
}

function renderHistoryLog() {
  if (!$historyList || !$historyEmptyState) return;

  if (historyLogEntries.length === 0) {
    $historyEmptyState.style.display = "flex";
    $historyList.style.display = "none";
    if ($historyCountBadge) $historyCountBadge.textContent = "0 events";
    return;
  }

  $historyEmptyState.style.display = "none";
  $historyList.style.display = "block";
  if ($historyCountBadge) $historyCountBadge.textContent = `${historyLogEntries.length} events`;

  $historyList.innerHTML = "";
  historyLogEntries.forEach((item) => {
    const li = document.createElement("li");
    li.className = "history-item";

    const isAction = ["SPACE", "COMMIT", "CLEAR", "BACKSPACE", "RESET"].includes(item.glyph);
    const glyphClass = isAction ? "glyph-action" : "glyph-letter";

    li.innerHTML = `
      <div class="hist-glyph-badge ${glyphClass}">
        ${escapeHtml(item.glyph)}
      </div>
      <div class="hist-info">
        <div class="hist-top-line">
          <span class="hist-type">${escapeHtml(item.type)}</span>
          <span class="hist-time">${escapeHtml(item.timestamp)}</span>
        </div>
        ${
          item.score !== undefined
            ? `<div class="hist-conf-line">
                 <span class="hist-conf-label">Confidence:</span>
                 <span class="hist-conf-val">${item.score}%</span>
               </div>`
            : ""
        }
      </div>
    `;
    $historyList.appendChild(li);
  });
}

function clearHistoryLog() {
  historyLogEntries = [];
  renderHistoryLog();
  showToast("History log cleared.", "info");
}

function formatTimestamp(d) {
  const pad = (n) => String(n).padStart(2, "0");
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}

function escapeHtml(str) {
  return String(str).replace(/[&<>"']/g, (m) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  }[m]));
}

// ------------------------------------------------------------------ //
//  Language Selection & Translation Triggers                         //
// ------------------------------------------------------------------ //

async function setLanguage(lang) {
  try {
    const res = await fetch(getApiUrl("/api/language"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ language: lang }),
    });
    const data = await res.json();
    if (data.ok) {
      activeLang = data.language;
      setActiveLanguageUI(activeLang);
      if ($alphaLangName) $alphaLangName.textContent = activeLang;
      updateVoiceBadgeUI(activeLang);
      fetchCurrentTranslation();
      showToast(`Language switched to ${activeLang}`, "info");
    }
  } catch (err) {
    console.error("[Language] Failed to switch language:", err);
    showToast("Language switch failed.", "error");
  }
}

let translationRequestSeq = 0;
let lastTranslatedText = "";
let lastTranslatedLang = "";

async function fetchCurrentTranslation() {
  const sentence = ($alphaSentenceText?.textContent || "").replace(/^—$/, "").trim();
  const word = ($alphaCurrentWord?.textContent || "").replace(/Awaiting.*/, "").trim();
  const textToTranslate = sentence || word;

  if (!textToTranslate) {
    lastTranslatedText = "";
    lastTranslatedLang = "";
    if ($alphaTranslatedText) {
      $alphaTranslatedText.textContent = "—";
      $alphaTranslatedText.classList.add("empty");
      $alphaTranslatedText.classList.remove("translating");
    }
    return;
  }

  // Avoid redundant in-flight or duplicate requests for exact same text and language
  if (
    textToTranslate === lastTranslatedText &&
    activeLang === lastTranslatedLang &&
    $alphaTranslatedText &&
    !$alphaTranslatedText.classList.contains("empty") &&
    $alphaTranslatedText.textContent !== "—"
  ) {
    return;
  }

  const reqSeq = ++translationRequestSeq;

  // Visual loading indicator for in-flight translation
  if ($alphaTranslatedText) {
    $alphaTranslatedText.classList.add("translating");
  }

  try {
    const res = await fetch(getApiUrl("/api/translate"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: textToTranslate,
        language: activeLang,
      }),
    });

    // Guard against stale responses: ignore if newer request was dispatched
    if (reqSeq < translationRequestSeq) {
      return;
    }

    if (res.ok) {
      const data = await res.json();
      if (reqSeq < translationRequestSeq) return;

      if (data.translation !== undefined && $alphaTranslatedText) {
        lastTranslatedText = textToTranslate;
        lastTranslatedLang = activeLang;
        const translatedStr = data.translation.trim();
        $alphaTranslatedText.textContent = translatedStr || "—";
        $alphaTranslatedText.classList.toggle("empty", !translatedStr);

        // Informative engine tooltip without layout shifts
        if (data.engine === "indictrans2") {
          $alphaTranslatedText.setAttribute("title", "Neural Machine Translation (IndicTrans2)");
        } else if (data.engine === "legacy") {
          $alphaTranslatedText.setAttribute("title", "Local Dictionary Translation");
        }
      }
    } else {
      // Fallback to legacy endpoint
      const legacyRes = await fetch(getApiUrl("/api/alphabet/translation"));
      if (reqSeq < translationRequestSeq) return;

      if (legacyRes.ok) {
        const legacyData = await legacyRes.json();
        if (legacyData.translation !== undefined && $alphaTranslatedText) {
          lastTranslatedText = textToTranslate;
          lastTranslatedLang = activeLang;
          const legacyStr = legacyData.translation.trim();
          $alphaTranslatedText.textContent = legacyStr || "—";
          $alphaTranslatedText.classList.toggle("empty", !legacyStr);
        }
      }
    }
  } catch (err) {
    console.warn("[Translation] Fetch error:", err);
  } finally {
    if (reqSeq === translationRequestSeq && $alphaTranslatedText) {
      $alphaTranslatedText.classList.remove("translating");
    }
  }
}

function setActiveLanguageUI(lang) {
  $langButtons.forEach((btn) => {
    const isTarget = btn.getAttribute("data-lang") === lang;
    btn.classList.toggle("active", isTarget);
    btn.setAttribute("aria-pressed", isTarget ? "true" : "false");
  });
}

// ------------------------------------------------------------------ //
//  WordBuffer Actions API                                            //
// ------------------------------------------------------------------ //

const BUFFER_ENDPOINT_MAP = {
  space: "/api/alphabet/space",
  backspace: "/api/alphabet/backspace",
  commit: "/api/alphabet/commit",
  clear_word: "/api/alphabet/clear-word",
  clear_sentence: "/api/alphabet/clear-sentence",
  clear_all: "/api/alphabet/clear-all",
  reset: "/api/alphabet/reset",
};

async function postBufferAction(action) {
  const url = getApiUrl(BUFFER_ENDPOINT_MAP[action] || `/api/buffer/${action}`);
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });
    const data = await res.json();

    if (data.sentence !== undefined) {
      if ($alphaSentenceText) {
        $alphaSentenceText.textContent = data.sentence.trim() || "—";
        $alphaSentenceText.classList.toggle("empty", !data.sentence.trim());
      }
    }
    if (data.current_word !== undefined) {
      if ($alphaCurrentWord) {
        if (data.current_word.trim()) {
          $alphaCurrentWord.textContent = data.current_word.trim();
          $alphaCurrentWord.classList.remove("empty");
        } else {
          $alphaCurrentWord.innerHTML = '<span class="word-placeholder">Awaiting fingerspelled letters...</span>';
          $alphaCurrentWord.classList.add("empty");
        }
      }
    }

    fetchCurrentTranslation();

    const actionLabelMap = {
      space: "SPACE",
      backspace: "BKSP",
      commit: "COMMIT",
      clear_word: "CLR WORD",
      clear_sentence: "CLR SENT",
      clear_all: "CLEAR ALL",
      reset: "RESET",
    };

    recordHistoryEntry({
      type: "USER ACTION",
      glyph: actionLabelMap[action] || action.toUpperCase(),
      timestamp: formatTimestamp(new Date()),
    });
  } catch (err) {
    console.error(`[Buffer] Action ${action} failed:`, err);
    showToast(`Action ${action} failed.`, "error");
  }
}

// ------------------------------------------------------------------ //
//  Manual Sign Commitment (+ Commit Sign)                           //
// ------------------------------------------------------------------ //

async function commitCurrentSign() {
  const letter = (latestConfirmedLetter !== "--" ? latestConfirmedLetter : latestRawPrediction).trim().toUpperCase();
  if (!letter || letter === "--" || letter === "...") {
    showToast("No active sign detected to append.", "warning");
    return;
  }

  try {
    const res = await fetch(getApiUrl("/api/alphabet/commit-letter"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ letter: letter }),
    });
    const data = await res.json();
    if (data.ok) {
      showToast(`Appended letter "${letter}" to word buffer`, "success");
      if (data.current_word !== undefined && $alphaCurrentWord) {
        $alphaCurrentWord.textContent = data.current_word;
        $alphaCurrentWord.classList.remove("empty");
      }
      fetchCurrentTranslation();
      recordHistoryEntry({
        type: "MANUAL COMMIT",
        glyph: letter,
        score: Math.round(latestConfidence * 100),
        timestamp: formatTimestamp(new Date()),
      });
    } else {
      postBufferAction("commit");
    }
  } catch (err) {
    postBufferAction("commit");
  }
}

// ------------------------------------------------------------------ //
//  Server-Side Fallback Speech Trigger                               //
// ------------------------------------------------------------------ //

async function triggerServerSpeech(type = "translated") {
  const rawUrl = type === "translated" ? "/api/alphabet/speak-translation" : "/api/alphabet/speak";
  try {
    await fetch(getApiUrl(rawUrl), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ type: type }),
    });
  } catch (err) {
    console.warn("[Server Speech Warning]", err);
  }
}

// ------------------------------------------------------------------ //
//  Settings Modal & Preferences Management                           //
// ------------------------------------------------------------------ //

function openSettingsModal() {
  if (!$modalSettings) return;
  const currentUrl = localStorage.getItem("isl_backend_url") || "";
  if ($settingBackendUrl) $settingBackendUrl.value = currentUrl;
  if ($settingMirrorToggle) $settingMirrorToggle.checked = isMirrored;
  const $neuralToggle = document.getElementById("setting-neural-tts-toggle");
  if ($neuralToggle) $neuralToggle.checked = isNeuralTTSEnabled;
  if ($settingConfThreshold) {
    $settingConfThreshold.value = Math.round(userConfThreshold * 100);
    if ($settingConfVal) $settingConfVal.textContent = `${$settingConfThreshold.value}%`;
  }
  if ($settingFpsTarget) {
    $settingFpsTarget.value = userTargetFps;
    if ($settingFpsVal) $settingFpsVal.textContent = `${userTargetFps} FPS`;
  }
  if ($pingStatusRow) $pingStatusRow.style.display = "none";
  fetchLearningStats();
  loadLearningPreferences();
  $modalSettings.style.display = "flex";
}

function closeSettingsModal() {
  if ($modalSettings) $modalSettings.style.display = "none";
}

async function testBackendPing() {
  if (!$pingStatusRow || !$pingDot || !$pingText) return;
  $pingStatusRow.style.display = "flex";
  $pingDot.className = "ping-dot pinging";
  $pingText.textContent = "Testing network latency...";

  const targetUrl = ($settingBackendUrl?.value || "").trim();
  const testUrl = targetUrl ? `${targetUrl.replace(/\/+$/, "")}/api/healthz` : getApiUrl("/api/healthz");

  const t0 = performance.now();
  try {
    const res = await fetch(testUrl, { method: "GET", cache: "no-store" });
    const elapsed = Math.round(performance.now() - t0);
    if (res.ok) {
      $pingDot.className = "ping-dot ok";
      $pingText.textContent = `Online — Latency: ${elapsed}ms (Status: 200 OK)`;
    } else {
      $pingDot.className = "ping-dot err";
      $pingText.textContent = `Server responded with status ${res.status} (${elapsed}ms)`;
    }
  } catch (err) {
    $pingDot.className = "ping-dot err";
    $pingText.textContent = `Connection failed: ${err.message}`;
  }
}

function saveSettings() {
  const newUrl = ($settingBackendUrl?.value || "").trim();
  if (newUrl) {
    localStorage.setItem("isl_backend_url", newUrl);
  } else {
    localStorage.removeItem("isl_backend_url");
  }

  if ($settingMirrorToggle) {
    isMirrored = $settingMirrorToggle.checked;
    localStorage.setItem("isl_mirror", isMirrored ? "1" : "0");
    syncMirroring();
  }

  const $neuralToggle = document.getElementById("setting-neural-tts-toggle");
  if ($neuralToggle) {
    isNeuralTTSEnabled = $neuralToggle.checked;
    localStorage.setItem("isl_neural_tts", isNeuralTTSEnabled ? "1" : "0");
  }

  if ($settingConfThreshold) {
    userConfThreshold = parseInt($settingConfThreshold.value, 10) / 100;
    localStorage.setItem("isl_conf_threshold", String(userConfThreshold));
  }

  if ($settingFpsTarget) {
    userTargetFps = parseInt($settingFpsTarget.value, 10);
    localStorage.setItem("isl_target_fps", String(userTargetFps));
  }

  closeSettingsModal();
  showToast("Settings saved successfully.", "success");
}

function resetSettings() {
  localStorage.removeItem("isl_backend_url");
  localStorage.removeItem("isl_mirror");
  localStorage.removeItem("isl_conf_threshold");
  localStorage.removeItem("isl_target_fps");
  localStorage.removeItem("isl_neural_tts");
  localStorage.removeItem("isl_feedback_session_token");
  sessionStorage.removeItem("isl_feedback_session_token");
  isMirrored = true;
  userConfThreshold = 0.60;
  userTargetFps = 12;
  isNeuralTTSEnabled = false;

  if ($settingBackendUrl) $settingBackendUrl.value = "";
  if ($settingMirrorToggle) $settingMirrorToggle.checked = true;
  const $neuralToggle = document.getElementById("setting-neural-tts-toggle");
  if ($neuralToggle) $neuralToggle.checked = false;
  if ($settingConfThreshold) {
    $settingConfThreshold.value = 60;
    if ($settingConfVal) $settingConfVal.textContent = "60%";
  }
  if ($settingFpsTarget) {
    $settingFpsTarget.value = 12;
    if ($settingFpsVal) $settingFpsVal.textContent = "12 FPS";
  }
  syncMirroring();
  showToast("Settings reset to default values.", "info");
}

function syncMirroring() {
  const videoEl = $browserWebcamVideo || document.getElementById("browser-webcam-video");
  if (videoEl) {
    videoEl.classList.toggle("mirrored", isMirrored);
    videoEl.classList.toggle("unmirrored", !isMirrored);
    videoEl.style.transform = isMirrored ? "scaleX(-1)" : "scaleX(1)";
  }
  if ($landmarkCanvas) {
    // Canvas receives normalized coordinates matching the active mirrored viewport.
    // We explicitly avoid applying scaleX(-1) to canvas to prevent double-inversion.
    $landmarkCanvas.classList.toggle("mirrored", false);
    $landmarkCanvas.classList.toggle("unmirrored", true);
    $landmarkCanvas.style.transform = "none";
  }
}

function loadPersistedSettings() {
  const savedMirror = localStorage.getItem("isl_mirror");
  if (savedMirror !== null) {
    isMirrored = savedMirror === "1";
  }
  const savedConf = localStorage.getItem("isl_conf_threshold");
  if (savedConf !== null) {
    userConfThreshold = parseFloat(savedConf) || 0.60;
  }
  const savedFps = localStorage.getItem("isl_target_fps");
  if (savedFps !== null) {
    userTargetFps = parseInt(savedFps, 10) || 12;
  }
  const savedNeural = localStorage.getItem("isl_neural_tts");
  if (savedNeural !== null) {
    isNeuralTTSEnabled = savedNeural === "1";
  }
  const $neuralToggle = document.getElementById("setting-neural-tts-toggle");
  if ($neuralToggle) {
    $neuralToggle.checked = isNeuralTTSEnabled;
  }
  syncMirroring();
}

// ------------------------------------------------------------------ //
//  Shortcuts Modal Handlers                                          //
// ------------------------------------------------------------------ //

function openShortcutsModal() {
  if ($modalShortcuts) $modalShortcuts.style.display = "flex";
}

function closeShortcutsModal() {
  if ($modalShortcuts) $modalShortcuts.style.display = "none";
}

function toggleHudOverlay() {
  hudVisible = !hudVisible;
  if ($cameraHud) {
    $cameraHud.style.display = hudVisible ? "flex" : "none";
  }
  if ($btnToggleHud) {
    $btnToggleHud.textContent = `HUD: ${hudVisible ? "ON" : "OFF"}`;
    $btnToggleHud.classList.toggle("active", hudVisible);
  }
}

function reconnectCameraStream() {
  if (isCloudMode) {
    showToast("Re-initializing browser webcam...", "info");
    if (browserWebcamStream) {
      browserWebcamStream.getTracks().forEach((t) => {
        try { t.stop(); } catch (e) {}
      });
      browserWebcamStream = null;
    }
    streamingActive = false;
    enableCloudWebcamMode();
    return;
  }
  if (!$cameraImg) return;
  showToast("Reconnecting video feed...", "info");
  const timestamp = Date.now();
  $cameraImg.src = getApiUrl(`/video_feed?t=${timestamp}`);
  if ($cameraOverlay) $cameraOverlay.style.display = "none";
}

function copySentenceToClipboard() {
  if (!$alphaSentenceText) return;
  const text = $alphaSentenceText.textContent;
  if (!text || text === "—") {
    showToast("Sentence is empty.", "info");
    return;
  }

  navigator.clipboard.writeText(text.trim()).then(
    () => showToast("Sentence copied to clipboard!", "success"),
    () => showToast("Clipboard access denied.", "error")
  );
}

// ------------------------------------------------------------------ //
//  Event Listeners & Keybindings                                      //
// ------------------------------------------------------------------ //

function setupEventListeners() {
  // WordBuffer Buttons
  if ($btnAlphaSpace) $btnAlphaSpace.addEventListener("click", () => postBufferAction("space"));
  if ($btnAlphaBackspace) $btnAlphaBackspace.addEventListener("click", () => postBufferAction("backspace"));
  if ($btnAlphaCommit) $btnAlphaCommit.addEventListener("click", () => postBufferAction("commit"));
  if ($btnAlphaClearWord) $btnAlphaClearWord.addEventListener("click", () => postBufferAction("clear_word"));
  if ($btnAlphaClearSent) $btnAlphaClearSent.addEventListener("click", () => postBufferAction("clear_all"));
  if ($btnAlphaReset) $btnAlphaReset.addEventListener("click", () => postBufferAction("reset"));

  // Manual Commit Sign
  if ($btnCommitSign) $btnCommitSign.addEventListener("click", commitCurrentSign);

  // Translation & Speech Actions
  if ($btnAlphaTranslate) $btnAlphaTranslate.addEventListener("click", fetchCurrentTranslation);
  if ($btnAlphaSpeakTrans) {
    $btnAlphaSpeakTrans.addEventListener("click", () => {
      const trans = ($alphaTranslatedText?.textContent || "").replace(/^—$/, "").trim();
      if (trans) {
        speakTextClient(trans, activeLang);
      } else {
        fetchCurrentTranslation().then(() => {
          const updated = ($alphaTranslatedText?.textContent || "").replace(/^—$/, "").trim();
          if (updated) speakTextClient(updated, activeLang);
        });
      }
    });
  }
  if ($btnAlphaSpeakOrig) {
    $btnAlphaSpeakOrig.addEventListener("click", () => {
      const sent = ($alphaSentenceText?.textContent || "").replace(/^—$/, "").trim();
      const word = ($alphaCurrentWord?.textContent || "").replace(/Awaiting.*/, "").trim();
      const text = sent || word;
      if (text) {
        speakTextClient(text, "English");
      } else {
        showToast("Sentence is empty.", "info");
      }
    });
  }
  if ($btnAlphaStopSpeech) $btnAlphaStopSpeech.addEventListener("click", stopSpeechClient);
  if ($btnAlphaReplaySpeech) $btnAlphaReplaySpeech.addEventListener("click", replaySpeechClient);

  // Copy Sentence Button
  if ($btnCopySentence) $btnCopySentence.addEventListener("click", copySentenceToClipboard);

  // HUD & Stream Reconnect Buttons
  if ($btnToggleHud) $btnToggleHud.addEventListener("click", toggleHudOverlay);
  if ($btnReconnectStream) $btnReconnectStream.addEventListener("click", reconnectCameraStream);

  // Safe Camera Session Toggle
  if ($btnSessionToggle) $btnSessionToggle.addEventListener("click", toggleCameraSession);

  // History Clear Button
  if ($btnClearHistory) $btnClearHistory.addEventListener("click", clearHistoryLog);

  // Shortcuts Modal Buttons
  if ($btnToggleShortcuts) $btnToggleShortcuts.addEventListener("click", openShortcutsModal);
  if ($btnCloseShortcuts) $btnCloseShortcuts.addEventListener("click", closeShortcutsModal);
  if ($btnDismissShortcuts) $btnDismissShortcuts.addEventListener("click", closeShortcutsModal);
  if ($modalShortcuts) {
    $modalShortcuts.addEventListener("click", (e) => {
      if (e.target === $modalShortcuts) closeShortcutsModal();
    });
  }

  // Settings Modal Buttons & Controls
  if ($btnBackendSettings) $btnBackendSettings.addEventListener("click", openSettingsModal);
  if ($btnCloseSettings) $btnCloseSettings.addEventListener("click", closeSettingsModal);
  if ($btnSaveSettings) $btnSaveSettings.addEventListener("click", saveSettings);
  if ($btnResetSettings) $btnResetSettings.addEventListener("click", resetSettings);
  if ($btnPingBackend) $btnPingBackend.addEventListener("click", testBackendPing);
  if ($modalSettings) {
    $modalSettings.addEventListener("click", (e) => {
      if (e.target === $modalSettings) closeSettingsModal();
    });
  }
  if ($settingConfThreshold) {
    $settingConfThreshold.addEventListener("input", () => {
      if ($settingConfVal) $settingConfVal.textContent = `${$settingConfThreshold.value}%`;
    });
  }
  if ($settingFpsTarget) {
    $settingFpsTarget.addEventListener("input", () => {
      if ($settingFpsVal) $settingFpsVal.textContent = `${$settingFpsTarget.value} FPS`;
    });
  }

  // Language Selection Buttons
  $langButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const lang = btn.getAttribute("data-lang");
      if (lang) setLanguage(lang);
    });
  });

  // Global Keyboard Shortcuts
  window.addEventListener("keydown", (e) => {
    if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName)) return;

    const key = e.key.toUpperCase();

    if (e.code === "Space") {
      e.preventDefault();
      postBufferAction("space");
    } else if (e.code === "Backspace") {
      e.preventDefault();
      postBufferAction("backspace");
    } else if (e.code === "Enter") {
      e.preventDefault();
      postBufferAction("commit");
    } else if (key === "T") {
      e.preventDefault();
      const trans = ($alphaTranslatedText?.textContent || "").replace(/^—$/, "").trim();
      if (trans) speakTextClient(trans, activeLang);
    } else if (key === "S") {
      e.preventDefault();
      const sent = ($alphaSentenceText?.textContent || "").replace(/^—$/, "").trim();
      if (sent) speakTextClient(sent, "English");
    } else if (key === "C") {
      e.preventDefault();
      postBufferAction("clear_word");
    } else if (key === "X") {
      e.preventDefault();
      postBufferAction("clear_all");
    } else if (key === "R") {
      e.preventDefault();
      postBufferAction("reset");
    } else if (key === "ESCAPE") {
      closeShortcutsModal();
      closeSettingsModal();
    } else if (KEY_TO_LANG[e.key]) {
      e.preventDefault();
      setLanguage(KEY_TO_LANG[e.key]);
    }
  });

  // Keep-alive heartbeat every 5s
  setInterval(() => {
    try {
      fetch(getApiUrl("/api/heartbeat"), { method: "POST" }).catch(() => {});
    } catch (e) {}
  }, 5000);
}

// ------------------------------------------------------------------ //
//  Development Mock Loop (Isolated behind DEV_MOCK_MODE flag)         //
// ------------------------------------------------------------------ //

function runDevMockLoop() {
  console.warn("[DEV] Running in DEV_MOCK_MODE. Synthetic data active.");
  updateSseStatus(true, "DEV MOCK");
  const letters = ["H", "E", "L", "L", "O", "W", "O", "R", "L", "D"];
  let idx = 0;
  let simulatedWord = "";

  setInterval(() => {
    const letter = letters[idx % letters.length];
    idx++;
    simulatedWord += letter;
    if (simulatedWord.length > 5) simulatedWord = letter;

    applyState({
      app_mode: "alphabet",
      camera_ok: true,
      model_ok: true,
      fps: 29.4,
      speech_status: "Ready",
      alphabet_hand_detected: true,
      alphabet_hand_count: 1,
      alphabet_hand_status: "1 HAND",
      alphabet_raw_letter: letter,
      alphabet_confidence: 0.88,
      alphabet_stable_letter: letter,
      alphabet_emitted_letter: letter,
      alphabet_current_word: simulatedWord,
      alphabet_sentence: "HELLO WORLD ",
      alphabet_translation: "नमस्ते दुनिया",
      alphabet_state: "EMITTED",
      stability_progress: 1.0,
      last_stable_letter: letter,
    });
  }, 1500);
}

// ------------------------------------------------------------------ //
//  Continuous Learning & Feedback Subsystem                           //
// ------------------------------------------------------------------ //

let currentFeedbackPredictionId = null;
let currentFeedbackLetter = null;
let lastPromptedLetter = null;
let lastPromptedTime = 0;
let feedbackDismissTimer = null;
let isFeedbackOpen = false;
let latestPredictionId = null;
let latestPredictedLetter = "--";
let selectedCorrectionLetter = null;
let isSessionTerminated = false;

function initFeedbackWidget() {
  const $grid = document.getElementById("feedback-az-grid");
  if ($grid && $grid.children.length === 0) {
    const letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");
    letters.forEach((char) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "btn-az-choice";
      btn.textContent = char;
      btn.addEventListener("click", () => submitCorrection(char));
      $grid.appendChild(btn);
    });
  }

  const $btnYes = document.getElementById("btn-feedback-yes");
  const $btnNo = document.getElementById("btn-feedback-no");
  const $btnSkip = document.getElementById("btn-feedback-skip");
  const $btnDismiss = document.getElementById("btn-feedback-dismiss");

  if ($btnYes) $btnYes.addEventListener("click", confirmPrediction);
  if ($btnNo) $btnNo.addEventListener("click", showAlphabetPicker);
  if ($btnSkip) $btnSkip.addEventListener("click", dismissFeedbackWidget);
  if ($btnDismiss) $btnDismiss.addEventListener("click", dismissFeedbackWidget);

  // Bind settings toggles and actions
  const $persToggle = document.getElementById("setting-personal-learning-toggle");
  const $globToggle = document.getElementById("setting-global-optin-toggle");
  const $btnResetPers = document.getElementById("btn-reset-personal-learning");

  if ($persToggle) {
    $persToggle.addEventListener("change", () => {
      updateLearningPreferences({ personalization_enabled: $persToggle.checked });
    });
  }

  if ($globToggle) {
    $globToggle.addEventListener("change", () => {
      if ($globToggle.checked) {
        fetch(getApiUrl("/api/v2/learning/global-consent"), {
          method: "POST",
          headers: getFeedbackHeaders(),
        })
          .then((r) => {
            checkResponseSessionToken(r);
            return r.json();
          })
          .then((d) => {
            showToast("Global dataset contribution enabled. Thank you!", "info");
            fetchLearningStats();
          })
          .catch((e) => console.error("Consent error:", e));
      } else {
        fetch(getApiUrl("/api/v2/learning/global-consent"), {
          method: "DELETE",
          headers: getFeedbackHeaders(),
        })
          .then((r) => {
            checkResponseSessionToken(r);
            return r.json();
          })
          .then((d) => {
            showToast("Global dataset contribution disabled.", "info");
            fetchLearningStats();
          })
          .catch((e) => console.error("Consent revocation error:", e));
      }
    });
  }

  if ($btnResetPers) {
    $btnResetPers.addEventListener("click", () => {
      if (confirm("Are you sure you want to erase all your personal learning corrections? This cannot be undone.")) {
        fetch(getApiUrl("/api/v2/learning/personal-data"), {
          method: "DELETE",
          headers: getFeedbackHeaders(),
        })
          .then((r) => {
            checkResponseSessionToken(r);
            return r.json();
          })
          .then((d) => {
            showToast("Personal learning data successfully erased.", "info");
            fetchLearningStats();
          })
          .catch((e) => console.error("Reset error:", e));
      }
    });
  }
}

function handleFeedbackOpportunity(data) {
  if (data && data.session_token) {
    saveFeedbackSessionToken(data.session_token);
  }
  if (isFeedbackOpen) return;

  const letter = data.last_stable_letter || data.fused_prediction || data.prediction;
  const predId = data.prediction_id;
  const conf = data.fused_confidence || data.confidence || 0.0;

  if (!letter || letter === "--" || letter === "None" || letter.length !== 1) return;
  if (!predId) return;

  const now = Date.now();
  // Don't ask about the same letter within 12 seconds, or if recently dismissed
  if (letter === lastPromptedLetter && now - lastPromptedTime < 12000) return;
  if (now - lastPromptedTime < 8000) return;

  showFeedbackWidget(letter, conf, predId);
}

function showFeedbackWidget(letter, confidence, predictionId) {
  currentFeedbackPredictionId = predictionId;
  currentFeedbackLetter = letter;
  lastPromptedLetter = letter;
  lastPromptedTime = Date.now();
  isFeedbackOpen = true;

  const $widget = document.getElementById("feedback-widget");
  const $predChar = document.getElementById("feedback-pred-char");
  const $confVal = document.getElementById("feedback-conf-val");
  const $initialActions = document.getElementById("feedback-initial-actions");
  const $azSelector = document.getElementById("feedback-alphabet-selector");
  const $ack = document.getElementById("feedback-acknowledgement");

  if (!$widget) return;

  if ($predChar) $predChar.textContent = letter;
  if ($confVal) $confVal.textContent = `${Math.round(confidence * 100)}%`;

  if ($initialActions) $initialActions.style.display = "flex";
  if ($azSelector) $azSelector.style.display = "none";
  if ($ack) $ack.style.display = "none";

  $widget.style.display = "block";

  // Auto-dismiss after 12 seconds if user ignores
  if (feedbackDismissTimer) clearTimeout(feedbackDismissTimer);
  feedbackDismissTimer = setTimeout(() => {
    dismissFeedbackWidget();
  }, 12000);
}

function confirmPrediction() {
  if (!currentFeedbackPredictionId) return;

  fetch(getApiUrl("/api/v2/feedback/confirm"), {
    method: "POST",
    headers: getFeedbackHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ prediction_id: currentFeedbackPredictionId }),
  })
    .then((r) => {
      checkResponseSessionToken(r);
      return r.json();
    })
    .then(() => {
      showFeedbackAcknowledgement("Prediction confirmed. Accuracy recorded!");
      fetchLearningStats();
      setTimeout(dismissFeedbackWidget, 2500);
    })
    .catch((err) => {
      console.warn("Feedback confirm error:", err);
      dismissFeedbackWidget();
    });
}

function showAlphabetPicker() {
  if (feedbackDismissTimer) clearTimeout(feedbackDismissTimer);
  const $initialActions = document.getElementById("feedback-initial-actions");
  const $azSelector = document.getElementById("feedback-alphabet-selector");

  if ($initialActions) $initialActions.style.display = "none";
  if ($azSelector) $azSelector.style.display = "block";
}

function submitCorrection(actualLetter) {
  if (!currentFeedbackPredictionId) return;

  fetch(getApiUrl("/api/v2/feedback/correct"), {
    method: "POST",
    headers: getFeedbackHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({
      prediction_id: currentFeedbackPredictionId,
      actual_label: actualLetter,
    }),
  })
    .then((r) => {
      checkResponseSessionToken(r);
      return r.json();
    })
    .then((res) => {
      const msg = res.message || "Correction saved. Thank you for helping improve recognition.";
      showFeedbackAcknowledgement(msg);
      fetchLearningStats();
      setTimeout(dismissFeedbackWidget, 3000);
    })
    .catch((err) => {
      console.warn("Correction submission error:", err);
      dismissFeedbackWidget();
    });
}

function showFeedbackAcknowledgement(text) {
  const $azSelector = document.getElementById("feedback-alphabet-selector");
  const $initialActions = document.getElementById("feedback-initial-actions");
  const $ack = document.getElementById("feedback-acknowledgement");
  const $ackText = document.getElementById("feedback-ack-text");

  if ($azSelector) $azSelector.style.display = "none";
  if ($initialActions) $initialActions.style.display = "none";
  if ($ackText) $ackText.textContent = text;
  if ($ack) $ack.style.display = "flex";
}

function dismissFeedbackWidget() {
  if (feedbackDismissTimer) {
    clearTimeout(feedbackDismissTimer);
    feedbackDismissTimer = null;
  }
  const $widget = document.getElementById("feedback-widget");
  if ($widget) $widget.style.display = "none";
  isFeedbackOpen = false;
  currentFeedbackPredictionId = null;
}

function fetchLearningStats() {
  fetch(getApiUrl("/api/v2/feedback/stats"), {
    headers: getFeedbackHeaders(),
  })
    .then((r) => {
      checkResponseSessionToken(r);
      return r.json();
    })
    .then((data) => {
      if (data && data.stats) {
        const s = data.stats;
        const $rev = document.getElementById("stat-reviewed-count");
        const $conf = document.getElementById("stat-confirmed-count");
        const $corr = document.getElementById("stat-corrected-count");
        const $glob = document.getElementById("stat-global-count");

        if ($rev) $rev.textContent = s.total_reviewed || 0;
        if ($conf) $conf.textContent = s.correct_confirmations || 0;
        if ($corr) $corr.textContent = s.incorrect_corrections || 0;
        if ($glob) $glob.textContent = s.global_contributions_total || 0;
      }
    })
    .catch(() => {});
}

function loadLearningPreferences() {
  fetch(getApiUrl("/api/v2/learning/preferences"), {
    headers: getFeedbackHeaders(),
  })
    .then((r) => {
      checkResponseSessionToken(r);
      return r.json();
    })
    .then((data) => {
      if (data && data.ok) {
        const $pers = document.getElementById("setting-personal-learning-toggle");
        const $glob = document.getElementById("setting-global-optin-toggle");
        if ($pers && data.personalization_enabled !== undefined) {
          $pers.checked = Boolean(data.personalization_enabled);
        }
        if ($glob && data.global_contribution_enabled !== undefined) {
          $glob.checked = Boolean(data.global_contribution_enabled);
        }
      }
    })
    .catch(() => {});
}

function updateLearningPreferences(prefs) {
  fetch(getApiUrl("/api/v2/learning/preferences"), {
    method: "PUT",
    headers: getFeedbackHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(prefs),
  })
    .then((r) => {
      checkResponseSessionToken(r);
      return r.json();
    })
    .then(() => fetchLearningStats())
    .catch(() => {});
}

// ------------------------------------------------------------------ //
//  Incorrect Prediction Correction Subsystem                         //
// ------------------------------------------------------------------ //

function initCorrectionModal() {
  const $grid = document.getElementById("correction-az-buttons");
  const $select = document.getElementById("correction-select-letter");

  if ($grid && $grid.children.length === 0) {
    const letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");
    letters.forEach((char) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "btn-az-btn";
      btn.dataset.char = char;
      btn.textContent = char;
      btn.setAttribute("aria-label", `Select letter ${char}`);
      btn.addEventListener("click", () => selectCorrectionLetter(char));
      $grid.appendChild(btn);
    });
  }

  if ($select && $select.children.length <= 1) {
    const letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");
    letters.forEach((char) => {
      const opt = document.createElement("option");
      opt.value = char;
      opt.textContent = `Letter ${char}`;
      $select.appendChild(opt);
    });
    $select.addEventListener("change", (e) => {
      if (e.target.value) {
        selectCorrectionLetter(e.target.value);
      }
    });
  }

  // Bind correction modal controls
  const $btnCorrect = document.getElementById("btn-correct-prediction");
  const $btnCloseCorr = document.getElementById("btn-close-correction");
  const $btnCancelCorr = document.getElementById("btn-cancel-correction");
  const $btnSubmitCorr = document.getElementById("btn-submit-correction");
  const $modalCorr = document.getElementById("modal-correction");

  if ($btnCorrect) $btnCorrect.addEventListener("click", openCorrectionModal);
  if ($btnCloseCorr) $btnCloseCorr.addEventListener("click", closeCorrectionModal);
  if ($btnCancelCorr) $btnCancelCorr.addEventListener("click", closeCorrectionModal);
  if ($btnSubmitCorr) $btnSubmitCorr.addEventListener("click", submitUserCorrection);
  if ($modalCorr) {
    $modalCorr.addEventListener("click", (e) => {
      if (e.target === $modalCorr) closeCorrectionModal();
    });
  }

  // Bind quit website modal controls
  const $btnQuit = document.getElementById("btn-quit-app");
  const $btnCloseQuit = document.getElementById("btn-close-quit-modal");
  const $btnCancelQuit = document.getElementById("btn-cancel-quit");
  const $btnConfirmQuit = document.getElementById("btn-confirm-quit");
  const $modalQuit = document.getElementById("modal-quit-confirm");
  const $btnReturnApp = document.getElementById("btn-return-to-app");

  if ($btnQuit) $btnQuit.addEventListener("click", openQuitConfirmModal);
  if ($btnCloseQuit) $btnCloseQuit.addEventListener("click", closeQuitConfirmModal);
  if ($btnCancelQuit) $btnCancelQuit.addEventListener("click", closeQuitConfirmModal);
  if ($btnConfirmQuit) $btnConfirmQuit.addEventListener("click", performCompleteSessionQuit);
  if ($modalQuit) {
    $modalQuit.addEventListener("click", (e) => {
      if (e.target === $modalQuit) closeQuitConfirmModal();
    });
  }
  if ($btnReturnApp) {
    $btnReturnApp.addEventListener("click", () => {
      window.location.reload();
    });
  }

  // Global ESC key listener for modals
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      closeCorrectionModal();
      closeQuitConfirmModal();
    }
  });
}

function openCorrectionModal() {
  if (isSessionTerminated) return;
  const $modal = document.getElementById("modal-correction");
  const $origGlyph = document.getElementById("correction-original-glyph");
  const $targetGlyph = document.getElementById("correction-target-glyph");
  const $predIdNote = document.getElementById("correction-pred-id-note");
  const $statusMsg = document.getElementById("correction-status-msg");
  const $submitBtn = document.getElementById("btn-submit-correction");
  const $select = document.getElementById("correction-select-letter");

  if (!$modal) return;

  selectedCorrectionLetter = null;
  if ($targetGlyph) $targetGlyph.textContent = "?";
  if ($submitBtn) {
    $submitBtn.disabled = true;
    $submitBtn.textContent = "Submit Correction";
  }
  if ($select) $select.value = "";

  document.querySelectorAll("#correction-az-buttons .btn-az-btn").forEach((b) => b.classList.remove("selected"));

  // Pre-fill original predicted sign (e.g. F)
  const predLetter = (latestPredictedLetter && latestPredictedLetter !== "--")
    ? latestPredictedLetter
    : (latestConfirmedLetter !== "--" ? latestConfirmedLetter : (latestRawPrediction || "--"));
  if ($origGlyph) $origGlyph.textContent = predLetter;

  const activePid = latestPredictionId || currentFeedbackPredictionId;
  if (activePid) {
    if ($predIdNote) $predIdNote.textContent = `Pred ID: ${activePid}`;
    if ($statusMsg) {
      $statusMsg.style.display = "none";
      $statusMsg.className = "correction-status-message";
      $statusMsg.textContent = "";
    }
  } else {
    if ($predIdNote) $predIdNote.textContent = "No prediction recorded yet";
    if ($statusMsg) {
      $statusMsg.style.display = "block";
      $statusMsg.className = "correction-status-message info";
      $statusMsg.textContent = "A prediction must first be recorded before it can be corrected. Perform a sign in front of the camera first.";
    }
  }

  $modal.style.display = "flex";
}

function closeCorrectionModal() {
  const $modal = document.getElementById("modal-correction");
  if ($modal) $modal.style.display = "none";
}

function selectCorrectionLetter(letter) {
  if (!letter || letter.length !== 1) return;
  selectedCorrectionLetter = letter.toUpperCase();

  const $targetGlyph = document.getElementById("correction-target-glyph");
  const $submitBtn = document.getElementById("btn-submit-correction");
  const $select = document.getElementById("correction-select-letter");

  if ($targetGlyph) $targetGlyph.textContent = selectedCorrectionLetter;
  if ($select) $select.value = selectedCorrectionLetter;

  document.querySelectorAll("#correction-az-buttons .btn-az-btn").forEach((b) => {
    b.classList.toggle("selected", b.dataset.char === selectedCorrectionLetter);
  });

  const activePid = latestPredictionId || currentFeedbackPredictionId;
  if ($submitBtn) {
    $submitBtn.disabled = !activePid || !selectedCorrectionLetter;
  }
}

async function submitUserCorrection() {
  const activePid = latestPredictionId || currentFeedbackPredictionId;
  const $submitBtn = document.getElementById("btn-submit-correction");
  const $statusMsg = document.getElementById("correction-status-msg");

  if (!activePid) {
    if ($statusMsg) {
      $statusMsg.style.display = "block";
      $statusMsg.className = "correction-status-message error";
      $statusMsg.textContent = "A prediction must first be recorded before it can be corrected.";
    }
    return;
  }

  if (!selectedCorrectionLetter) {
    if ($statusMsg) {
      $statusMsg.style.display = "block";
      $statusMsg.className = "correction-status-message error";
      $statusMsg.textContent = "Please select the correct sign (A–Z).";
    }
    return;
  }

  if ($submitBtn) {
    $submitBtn.disabled = true;
    $submitBtn.textContent = "Submitting...";
  }

  try {
    const res = await fetch(getApiUrl("/api/v2/feedback/correct"), {
      method: "POST",
      headers: getFeedbackHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({
        prediction_id: activePid,
        actual_label: selectedCorrectionLetter,
      }),
    });

    checkResponseSessionToken(res);
    const data = await res.json();

    if (res.ok && data.ok) {
      const origChar = document.getElementById("correction-original-glyph")?.textContent || "--";
      const successMsg = `Correction submitted: ${origChar} ➔ ${selectedCorrectionLetter}. Learning engine updated!`;
      if ($statusMsg) {
        $statusMsg.style.display = "block";
        $statusMsg.className = "correction-status-message success";
        $statusMsg.textContent = successMsg;
      }
      showToast(successMsg, "success");
      fetchLearningStats();

      // Update dashboard confirmed letter with user's verified correction
      latestConfirmedLetter = selectedCorrectionLetter;
      if ($alphaStableChar) {
        $alphaStableChar.textContent = selectedCorrectionLetter;
      }

      setTimeout(() => {
        closeCorrectionModal();
        if ($submitBtn) {
          $submitBtn.textContent = "Submit Correction";
        }
      }, 1600);
    } else {
      const errMsg = data.error || "Failed to submit correction.";
      if ($statusMsg) {
        $statusMsg.style.display = "block";
        $statusMsg.className = "correction-status-message error";
        $statusMsg.textContent = errMsg;
      }
      showToast(errMsg, "error");
      if ($submitBtn) {
        $submitBtn.disabled = false;
        $submitBtn.textContent = "Submit Correction";
      }
    }
  } catch (err) {
    const errMsg = err.message || "Network error submitting correction.";
    if ($statusMsg) {
      $statusMsg.style.display = "block";
      $statusMsg.className = "correction-status-message error";
      $statusMsg.textContent = errMsg;
    }
    showToast(errMsg, "error");
    if ($submitBtn) {
      $submitBtn.disabled = false;
      $submitBtn.textContent = "Submit Correction";
    }
  }
}

// ------------------------------------------------------------------ //
//  Quit Website Session Subsystem                                     //
// ------------------------------------------------------------------ //

function openQuitConfirmModal() {
  const $modal = document.getElementById("modal-quit-confirm");
  if ($modal) $modal.style.display = "flex";
}

function closeQuitConfirmModal() {
  const $modal = document.getElementById("modal-quit-confirm");
  if ($modal) $modal.style.display = "none";
}

function performCompleteSessionQuit() {
  closeQuitConfirmModal();

  // 1. Mark session as permanently terminated
  isSessionTerminated = true;
  streamingActive = false;

  // 2. Stop camera tracks, notify backend, and release media
  fetch(getApiUrl("/api/camera/stop"), { method: "POST" }).catch(() => {});
  if (streamLoopTimer) {
    clearTimeout(streamLoopTimer);
    streamLoopTimer = null;
  }
  if (browserWebcamStream) {
    browserWebcamStream.getTracks().forEach((t) => {
      try { t.stop(); } catch (e) {}
    });
    browserWebcamStream = null;
  }
  const videoEl = $browserWebcamVideo || document.getElementById("browser-webcam-video");
  if (videoEl) {
    videoEl.pause();
    videoEl.srcObject = null;
  }
  if ($cameraImg) {
    $cameraImg.src = "";
    $cameraImg.style.display = "none";
  }

  // 3. Invalidate in-flight inference responses
  monotonicFrameSeq += 100000;
  lastCommittedFrameSeq = monotonicFrameSeq;

  // 4. Cancel browser speech & audio
  try {
    if (window.speechSynthesis) {
      window.speechSynthesis.cancel();
    }
  } catch (e) {}

  // 5. Close SSE connection & network listeners
  disconnectSSE();
  clearTimeout(heartbeatTimer);
  clearTimeout(feedbackDismissTimer);

  // 6. Dismiss any open widgets/modals
  dismissFeedbackWidget();
  closeCorrectionModal();
  closeSettingsModal();
  closeShortcutsModal();

  // 7. Show fullscreen Exit Screen
  const $exitScreen = document.getElementById("exit-screen");
  if ($exitScreen) {
    $exitScreen.style.display = "flex";
  }

  // 8. Attempt tab close if permissible
  try {
    window.close();
  } catch (e) {}
}

// ------------------------------------------------------------------ //
//  Session Provisioning & Synchronization Subsystem                  //
// ------------------------------------------------------------------ //

async function initSessionToken() {
  try {
    const res = await fetch(getApiUrl("/api/v2/session"), {
      method: "POST",
      headers: getFeedbackHeaders({ "Content-Type": "application/json" }),
    });
    checkResponseSessionToken(res);
    if (res.ok) {
      const data = await res.json();
      if (data && data.session_token) {
        saveFeedbackSessionToken(data.session_token);
      }
    }
  } catch (err) {
    console.debug("[Session Init Warning]", err);
  }
}

// ------------------------------------------------------------------ //
//  Application Initialization                                         //
// ------------------------------------------------------------------ //

document.addEventListener("DOMContentLoaded", async () => {
  await initSessionToken();
  loadPersistedSettings();
  setupEventListeners();
  initFeedbackWidget();
  initCorrectionModal();
  loadVoiceCatalog();
  connectSSE();
  checkEnvironmentAndInitWebcam();

  // If voices list is async on Chrome, bind event
  if ("speechSynthesis" in window) {
    window.speechSynthesis.onvoiceschanged = () => {
      updateVoiceBadgeUI(activeLang);
    };
  }
});

// Safe cleanup on window unload
window.addEventListener("beforeunload", () => {
  disconnectSSE();
});
