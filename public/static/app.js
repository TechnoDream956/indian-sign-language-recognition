/**
 * ISL Translator — Frontend Application Logic (Dedicated Alphabet & Fingerspelling)
 *
 * Professional AI/Computer-Vision Workspace Frontend
 *
 * Features:
 *  - Real-time Server-Sent Events (SSE) listener (/events)
 *  - Live webcam stream health monitoring and reconnect handler
 *  - Real-time gesture classification, confidence meter, and hold-to-emit debouncer
 *  - Dual-hand landmark tracking telemetry and safety suppression display
 *  - WordBuffer assembly and sentence composition controls
 *  - Multilingual Indic translation and verified native TTS audio controls
 *  - Honest voice availability reporting (detects installed vs missing macOS voices)
 *  - In-session real-time recognition history log (zero synthetic data)
 *  - Keyboard shortcuts modal and global hotkeys
 *  - Clean application lifecycle, heartbeat, and graceful shutdown
 */

"use strict";

// ------------------------------------------------------------------ //
//  Development / Diagnostics Configuration                            //
// ------------------------------------------------------------------ //

const DEV_MOCK_MODE = false;

// ------------------------------------------------------------------ //
//  API & Backend Endpoint Configuration                             //
// ------------------------------------------------------------------ //

function getApiUrl(endpoint) {
  const base = window.ISL_BACKEND_URL || localStorage.getItem("isl_backend_url") || "";
  if (!base) return endpoint;
  const cleanBase = base.replace(/\/+$/, "");
  const cleanEndpoint = endpoint.startsWith("/") ? endpoint : `/${endpoint}`;
  return `${cleanBase}${cleanEndpoint}`;
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

// ------------------------------------------------------------------ //
//  DOM Element References                                            //
// ------------------------------------------------------------------ //

// Camera & Viewport
const $cameraImg            = document.getElementById("camera-img");
const $browserWebcamVideo   = document.getElementById("browser-webcam-video");
const $cameraHud            = document.getElementById("camera-hud");
const $btnToggleHud         = document.getElementById("btn-toggle-hud");
const $btnToggleCamera      = document.getElementById("btn-toggle-camera");
const $cameraOverlay        = document.getElementById("camera-overlay");
const $cameraOverlayMsg     = document.getElementById("camera-overlay-msg");
const $btnReconnectStream   = document.getElementById("btn-reconnect-stream");
const $cameraLoading        = document.getElementById("camera-loading");
const $liveIndicator        = document.getElementById("live-indicator");
const $hudModePill          = document.getElementById("hud-mode-pill");
const $fpsBadge             = document.getElementById("fps-badge");
const $hudFpsPill           = document.getElementById("hud-fps-pill");
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

// WordBuffer & Composer
const $alphaCurrentWord     = document.getElementById("alpha-current-word");
const $alphaSentenceText    = document.getElementById("alpha-sentence-text");
const $alphaWordCount       = document.getElementById("alpha-word-count");
const $btnCopySentence      = document.getElementById("btn-copy-sentence");

// Translation & Audio
const $alphaLangName        = document.getElementById("alpha-lang-name");
const $alphaVoiceBadge      = document.getElementById("alpha-voice-badge");
const $alphaTranslatedText  = document.getElementById("alpha-translated-text");
const $btnAlphaSpeakTrans   = document.getElementById("btn-alpha-speak-trans");
const $btnAlphaSpeakOrig    = document.getElementById("btn-alpha-speak-orig");
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

// Modals, Toast & Quit
const $btnToggleShortcuts   = document.getElementById("btn-toggle-shortcuts");
const $btnBackendSettings   = document.getElementById("btn-backend-settings");
const $modalShortcuts       = document.getElementById("modal-shortcuts");
const $btnCloseShortcuts    = document.getElementById("btn-close-shortcuts");
const $btnDismissShortcuts  = document.getElementById("btn-dismiss-shortcuts");
const $btnQuitApp           = document.getElementById("btn-quit-app");
const $appShutdownOverlay   = document.getElementById("app-shutdown-overlay");
const $toastContainer       = document.getElementById("toast-container");

// ------------------------------------------------------------------ //
//  Local Application State                                            //
// ------------------------------------------------------------------ //

let activeLang = "English";
let sseConnection = null;
let sseReconnectTimer = null;
let lastKnownEmittedLetter = null;
let lastProcessedEmissionId = null;
let historyLogEntries = [];
let hudVisible = true;
let voiceCatalog = {};

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
    console.warn("[TTS] Could not load voice catalog:", err);
  }
}

function updateVoiceBadgeUI(lang) {
  if (!$alphaVoiceBadge) return;
  const info = voiceCatalog[lang];
  if (!info) {
    $alphaVoiceBadge.textContent = "Voice: Status Unknown";
    $alphaVoiceBadge.className = "trans-voice-badge warn";
    return;
  }

  if (info.available && info.voice) {
    $alphaVoiceBadge.textContent = `Voice: ${info.voice} (${lang})`;
    $alphaVoiceBadge.className = "trans-voice-badge ok";
  } else if (lang === "English") {
    $alphaVoiceBadge.textContent = "Voice: English (System)";
    $alphaVoiceBadge.className = "trans-voice-badge ok";
  } else {
    $alphaVoiceBadge.textContent = `Voice: Not Installed in macOS`;
    $alphaVoiceBadge.className = "trans-voice-badge err";
  }
}

// ------------------------------------------------------------------ //
//  SSE — Real-Time State Listener                                      //
// ------------------------------------------------------------------ //

function connectSSE() {
  if (DEV_MOCK_MODE) {
    runDevMockLoop();
    return;
  }

  if (sseConnection) {
    try { sseConnection.close(); } catch (e) {}
  }

  updateSseStatus(false, "CONNECTING");

  try {
    sseConnection = new EventSource(getApiUrl("/events"));

    sseConnection.onopen = () => {
      updateSseStatus(true, "LIVE");
      if ($cameraOverlay && $cameraOverlay.dataset.source === "sse") {
        $cameraOverlay.style.display = "none";
      }
    };

    sseConnection.onmessage = (evt) => {
      try {
        const data = JSON.parse(evt.data);
        applyState(data);
      } catch (e) {
        console.warn("[SSE] Parse error on incoming packet:", e);
      }
    };

    sseConnection.onerror = () => {
      updateSseStatus(false, "RECONNECTING");
      try { sseConnection.close(); } catch (e) {}
      clearTimeout(sseReconnectTimer);
      sseReconnectTimer = setTimeout(connectSSE, 2500);
    };
  } catch (err) {
    console.error("[SSE] Connection initialization failed:", err);
    updateSseStatus(false, "ERROR");
    clearTimeout(sseReconnectTimer);
    sseReconnectTimer = setTimeout(connectSSE, 3000);
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
//  Cloud Browser Webcam Architecture                                 //
// ------------------------------------------------------------------ //

let isCloudMode = false;
let browserWebcamStream = null;
let browserWebcamStreamingInterval = null;
let isStreamingFrame = false;
let browserWebcamCanvas = null;

async function enableCloudWebcamMode() {
  if (isCloudMode && browserWebcamStream) return;
  isCloudMode = true;

  console.log("[Camera] Enabling Cloud Browser Webcam mode...");
  if ($cameraHintText) {
    $cameraHintText.textContent = "Cloud Mode: Streaming browser webcam frames to server inference engine";
  }
  if ($hudModePill) {
    $hudModePill.style.display = "inline-flex";
    $hudModePill.textContent = "CLOUD WEBCAM";
  }

  const videoEl = $browserWebcamVideo || document.getElementById("browser-webcam-video");
  if (!videoEl) {
    console.error("[Camera] browser-webcam-video element not found in DOM");
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

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        facingMode: "user"
      },
      audio: false
    });

    browserWebcamStream = stream;
    videoEl.srcObject = stream;
    await videoEl.play();
    videoEl.style.display = "block";

    if ($cameraImg) {
      $cameraImg.style.display = "none";
    }
    if ($cameraLoading) {
      $cameraLoading.style.display = "none";
    }
    if ($cameraOverlay) {
      $cameraOverlay.style.display = "none";
    }

    showToast("Webcam connected. Streaming frames to AI engine.", "success");
    startFrameStreaming(videoEl);
  } catch (err) {
    console.warn("[Camera] getUserMedia failed:", err);
    if ($cameraOverlay) {
      $cameraOverlay.style.display = "flex";
      if ($cameraOverlayMsg) {
        $cameraOverlayMsg.textContent = "Webcam access required in cloud mode. Click Reconnect to grant permission.";
      }
    }
    showToast("Webcam permission denied: " + err.message, "error");
  }
}

function startFrameStreaming(videoEl) {
  if (browserWebcamStreamingInterval) {
    clearInterval(browserWebcamStreamingInterval);
  }

  if (!browserWebcamCanvas) {
    browserWebcamCanvas = document.createElement("canvas");
    browserWebcamCanvas.width = 640;
    browserWebcamCanvas.height = 480;
  }
  const ctx = browserWebcamCanvas.getContext("2d");

  // Ingest frames every 80ms (~12.5 FPS)
  browserWebcamStreamingInterval = setInterval(() => {
    if (isStreamingFrame || !videoEl || videoEl.paused || videoEl.ended || !videoEl.videoWidth) {
      return;
    }
    isStreamingFrame = true;

    try {
      ctx.drawImage(videoEl, 0, 0, 640, 480);
      browserWebcamCanvas.toBlob(async (blob) => {
        if (!blob) {
          isStreamingFrame = false;
          return;
        }
        try {
          const res = await fetch(getApiUrl("/api/camera/frame"), {
            method: "POST",
            headers: { "Content-Type": "image/jpeg" },
            body: blob,
          });
          if (res.ok) {
            const data = await res.json();
            if (data && data.state) {
              applyState(data.state);
            }
          }
        } catch (postErr) {
          console.debug("[Frame Ingest Warning]", postErr);
        } finally {
          isStreamingFrame = false;
        }
      }, "image/jpeg", 0.7);
    } catch (e) {
      isStreamingFrame = false;
    }
  }, 80);
}

function stopFrameStreaming() {
  if (browserWebcamStreamingInterval) {
    clearInterval(browserWebcamStreamingInterval);
    browserWebcamStreamingInterval = null;
  }
  isStreamingFrame = false;
}

async function checkEnvironmentAndInitWebcam() {
  const urlParams = new URLSearchParams(window.location.search);
  const forceCloud = urlParams.get("mode") === "cloud" || window.FORCE_CLOUD_CAMERA === true;

  try {
    const res = await fetch(getApiUrl("/api/status"));
    if (res.ok) {
      const data = await res.json();
      if (data.cloud_mode || forceCloud) {
        enableCloudWebcamMode();
      }
    }
  } catch (err) {
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
  const isSpeaking = speechStatus === "Speaking...";
  if ($statusSpeech && $statusSpeechVal) {
    $statusSpeech.className = `telemetry-dot ${isSpeaking ? "pulse-active" : (speechStatus.includes("error") ? "err" : "ok")}`;
    $statusSpeechVal.textContent = speechStatus.toUpperCase();
  }
  if ($btnAlphaSpeakTrans) {
    $btnAlphaSpeakTrans.classList.toggle("speaking", isSpeaking);
  }
  if ($btnAlphaSpeakOrig) {
    $btnAlphaSpeakOrig.classList.toggle("speaking", isSpeaking);
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

  // Camera Toggle Button Sync
  if ($btnToggleCamera && state.camera_enabled !== undefined) {
    if (state.camera_enabled) {
      $btnToggleCamera.textContent = "📷 Pause Camera";
      $btnToggleCamera.classList.remove("paused");
    } else {
      $btnToggleCamera.textContent = "📷 Start Camera";
      $btnToggleCamera.classList.add("paused");
    }
  }

  // 7. Real-Time Gesture Recognition Display
  const rawLetter = state.fused_prediction || state.raw_prediction || state.prediction || state.alphabet_raw_letter || state.pred_label || "--";
  const isValidLetter = rawLetter !== "--" && rawLetter !== "...";

  if ($alphaBigChar) {
    $alphaBigChar.textContent = isValidLetter ? rawLetter : "--";
    $alphaBigChar.classList.toggle("detected", isValidLetter);
  }

  // Real Confidence Bar & Percentage
  const alphaConf = parseFloat(state.fused_confidence !== undefined ? state.fused_confidence : (state.alphabet_confidence !== undefined ? state.alphabet_confidence : (state.confidence || 0.0)));
  const alphaPct = Math.min(100, Math.max(0, Math.round(alphaConf * 100)));

  if ($alphaConfNum) {
    $alphaConfNum.textContent = isValidLetter ? `${alphaPct}%` : "0%";
  }
  if ($alphaConfBar) {
    $alphaConfBar.style.width = isValidLetter ? `${alphaPct}%` : "0%";
    if (alphaConf >= 0.70) {
      $alphaConfBar.style.background = "var(--color-primary-accent)";
    } else if (alphaConf >= 0.40) {
      $alphaConfBar.style.background = "var(--color-warm-accent)";
    } else {
      $alphaConfBar.style.background = "var(--color-alert)";
    }
  }

  // Stable Candidate & Debounce Telemetry
  const stableLetter = state.stable_letter || state.alphabet_stable_letter || "--";
  if ($alphaStableChar) {
    $alphaStableChar.textContent = stableLetter !== "--" ? stableLetter : "--";
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

  // Calibrate initial emission counter on idle connect without recording ghost entry
  if (lastProcessedEmissionId === null && emissionId && !currentEmitted) {
    lastProcessedEmissionId = emissionId;
  }

  // Record to history if a genuine new emission event occurred
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
    } else if (alphaState.includes("2 HANDS")) {
      badgeCategory = "idle";
    }
    $alphaStateBadge.className = `state-badge state-${badgeCategory}`;
  }

  if ($alphaLatchFill) {
    if (alphaState.startsWith("STABILIZING")) {
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

  // Word counter
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
//  Language Selection API                                            //
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

      // Immediately fetch updated translation for current sentence/word
      fetchCurrentTranslation();
      showToast(`Language switched to ${activeLang}`, "info");
    }
  } catch (err) {
    console.error("[Language] Failed to switch language:", err);
    showToast("Language switch failed.", "error");
  }
}

async function fetchCurrentTranslation() {
  try {
    const res = await fetch(getApiUrl("/api/alphabet/translation"));
    if (res.ok) {
      const data = await res.json();
      if (data.translation !== undefined && $alphaTranslatedText) {
        $alphaTranslatedText.textContent = data.translation.trim() || "—";
        $alphaTranslatedText.classList.toggle("empty", !data.translation.trim());
      }
    }
  } catch (err) {
    console.warn("[Translation] Instant fetch failed:", err);
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

    // Immediately update translation
    fetchCurrentTranslation();

    // Record action to history
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
//  Dedicated Speech Synthesis Triggers (English vs Translation)       //
// ------------------------------------------------------------------ //

async function triggerSpeech(type = "translated") {
  // Use dedicated endpoint to guarantee proper language and voice routing
  const rawUrl = type === "translated" ? "/api/alphabet/speak-translation" : "/api/alphabet/speak";
  const url = getApiUrl(rawUrl);
  
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ type: type }),
    });
    const data = await res.json();

    if (data.ok && data.triggered) {
      const $activeBtn = type === "translated" ? $btnAlphaSpeakTrans : $btnAlphaSpeakOrig;
      if ($activeBtn) {
        $activeBtn.classList.add("speaking");
        setTimeout(() => $activeBtn.classList.remove("speaking"), 2500);
      }

      if (type === "translated") {
        const voiceInfo = voiceCatalog[activeLang]?.voice || activeLang;
        showToast(`Speaking in ${activeLang} (${voiceInfo})...`, "success");
      } else {
        showToast("Speaking English sentence...", "info");
      }
    } else {
      // Diagnostic reason feedback
      const currentSentence = ($alphaSentenceText?.textContent || "").trim();
      const currentWord = ($alphaCurrentWord?.textContent || "").trim();
      const hasText = (currentSentence && currentSentence !== "—") || (currentWord && !currentWord.includes("Awaiting"));

      if (!hasText) {
        showToast("No sentence text available to speak.", "info");
      } else if (type === "translated" && ["Gujarati", "Malayalam", "Punjabi"].includes(activeLang)) {
        showToast(`Native voice for ${activeLang} is not installed in macOS. (Hindi, Bengali, Tamil, Telugu, and Kannada voices are installed)`, "warning");
      } else {
        showToast("Speech synthesis busy or voice unavailable.", "warning");
      }
    }
  } catch (err) {
    console.error("[Speech] Request failed:", err);
    showToast("Speech synthesis request failed.", "error");
  }
}

// ------------------------------------------------------------------ //
//  Copy Sentence to Clipboard                                        //
// ------------------------------------------------------------------ //

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
//  HUD Overlay Toggle                                                //
// ------------------------------------------------------------------ //

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

// ------------------------------------------------------------------ //
//  Camera Stream Reconnect                                           //
// ------------------------------------------------------------------ //

function reconnectCameraStream() {
  if (isCloudMode) {
    showToast("Re-initializing browser webcam...", "info");
    enableCloudWebcamMode();
    return;
  }
  if (!$cameraImg) return;
  showToast("Reconnecting video feed...", "info");
  const timestamp = Date.now();
  $cameraImg.src = getApiUrl(`/video_feed?t=${timestamp}`);
  if ($cameraOverlay) $cameraOverlay.style.display = "none";
}

// ------------------------------------------------------------------ //
//  Application Shutdown Coordinator                                  //
// ------------------------------------------------------------------ //

async function quitApplication() {
  if ($appShutdownOverlay) $appShutdownOverlay.style.display = "flex";
  try {
    await fetch(getApiUrl("/api/shutdown"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });
  } catch (err) {
    console.log("[Shutdown] Request emitted:", err);
  }
}

// ------------------------------------------------------------------ //
//  Keyboard Shortcuts Modal Handlers                                 //
// ------------------------------------------------------------------ //

function openShortcutsModal() {
  if ($modalShortcuts) $modalShortcuts.style.display = "flex";
}

function closeShortcutsModal() {
  if ($modalShortcuts) $modalShortcuts.style.display = "none";
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

  // Audio Buttons
  if ($btnAlphaSpeakTrans) $btnAlphaSpeakTrans.addEventListener("click", () => triggerSpeech("translated"));
  if ($btnAlphaSpeakOrig) $btnAlphaSpeakOrig.addEventListener("click", () => triggerSpeech("original"));

  // Copy Sentence Button
  if ($btnCopySentence) $btnCopySentence.addEventListener("click", copySentenceToClipboard);

  // HUD & Stream Reconnect Buttons
  if ($btnToggleHud) $btnToggleHud.addEventListener("click", toggleHudOverlay);
  if ($btnReconnectStream) $btnReconnectStream.addEventListener("click", reconnectCameraStream);
  if ($btnToggleCamera) {
    $btnToggleCamera.addEventListener("click", async () => {
      if (isCloudMode && browserWebcamStream) {
        const tracks = browserWebcamStream.getVideoTracks();
        const anyDisabled = tracks.some(t => !t.enabled);
        const willEnable = anyDisabled;
        tracks.forEach(t => t.enabled = willEnable);
        const videoEl = $browserWebcamVideo || document.getElementById("browser-webcam-video");
        if (willEnable) {
          if (videoEl && videoEl.paused) videoEl.play();
          startFrameStreaming(videoEl);
          showToast("Browser webcam feed resumed", "info");
          $btnToggleCamera.textContent = "📷 Pause Camera";
          $btnToggleCamera.classList.remove("paused");
        } else {
          stopFrameStreaming();
          if (videoEl) videoEl.pause();
          showToast("Browser webcam feed paused", "info");
          $btnToggleCamera.textContent = "📷 Resume Camera";
          $btnToggleCamera.classList.add("paused");
        }
        return;
      }
      try {
        const res = await fetch(getApiUrl("/api/camera/toggle"), { method: "POST" });
        const data = await res.json();
        if (data.ok) {
          showToast(data.camera_enabled ? "Camera resuming..." : "Camera paused", "info");
        }
      } catch (err) {
        showToast("Camera toggle failed", "error");
      }
    });
  }

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

  // Backend URL Settings Button
  if ($btnBackendSettings) {
    $btnBackendSettings.addEventListener("click", () => {
      const current = localStorage.getItem("isl_backend_url") || window.location.origin;
      const next = prompt("Configure Python Backend Server URL (e.g. https://my-isl-backend.onrender.com or http://localhost:5001):\n(Leave empty to use current host origin)", current);
      if (next !== null) {
        if (next.trim()) {
          localStorage.setItem("isl_backend_url", next.trim());
          showToast(`Backend URL set to: ${next.trim()}`, "info");
        } else {
          localStorage.removeItem("isl_backend_url");
          showToast("Backend URL reset to default host.", "info");
        }
        setTimeout(() => window.location.reload(), 800);
      }
    });
  }

  // Quit Application Button
  if ($btnQuitApp) $btnQuitApp.addEventListener("click", quitApplication);

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
      triggerSpeech("translated");
    } else if (key === "S") {
      e.preventDefault();
      triggerSpeech("original");
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
    } else if (key === "Q" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      quitApplication();
    } else if (KEY_TO_LANG[e.key]) {
      e.preventDefault();
      setLanguage(KEY_TO_LANG[e.key]);
    }
  });

  // Lifecycle Listeners: send beacon on tab/window close
  window.addEventListener("beforeunload", () => {
    try {
      navigator.sendBeacon(getApiUrl("/api/client-disconnect"));
    } catch (e) {}
  });

  window.addEventListener("pagehide", () => {
    try {
      navigator.sendBeacon(getApiUrl("/api/client-disconnect"));
    } catch (e) {}
  });

  // Heartbeat every 2.5 seconds to keep application alive while window is active
  setInterval(() => {
    try {
      fetch(getApiUrl("/api/heartbeat"), { method: "POST" }).catch(() => {});
    } catch (e) {}
  }, 2500);
}

// ------------------------------------------------------------------ //
//  Development Mock Loop (Isolated behind DEV_MOCK_MODE flag)         //
// ------------------------------------------------------------------ //

function runDevMockLoop() {
  console.warn("[DEV] Running in DEV_MOCK_MODE. Synthetic data is active.");
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
    });
  }, 1500);
}

// ------------------------------------------------------------------ //
//  Application Initialization                                         //
// ------------------------------------------------------------------ //

document.addEventListener("DOMContentLoaded", () => {
  setupEventListeners();
  loadVoiceCatalog();
  connectSSE();
  checkEnvironmentAndInitWebcam();
});
