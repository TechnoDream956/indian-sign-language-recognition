#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self-contained HTML/CSS/JS template for the Standalone Multilingual Translation & TTS Test Interface.
Operates completely independently without requiring webcam or sign recognition pipeline.
"""

TEST_PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ISL Multilingual Translation & Neural TTS Subsystem</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Noto+Sans+Devanagari:wght@500;700&family=Noto+Sans+Bengali:wght@500;700&family=Noto+Sans+Tamil:wght@500;700&family=Noto+Sans+Telugu:wght@500;700&family=Noto+Sans+Gujarati:wght@500;700&family=Noto+Sans+Kannada:wght@500;700&family=Noto+Sans+Malayalam:wght@500;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-card: #1e293b;
            --border-color: #334155;
            --accent: #38bdf8;
            --accent-hover: #0ea5e9;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --radius: 12px;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-main);
            min-height: 100vh;
            padding: 24px;
            display: flex;
            justify-content: center;
        }
        .container {
            width: 100%;
            max-width: 960px;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 16px;
        }
        .header h1 {
            font-size: 1.5rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 10px;
            color: #fff;
        }
        .badge {
            font-size: 0.75rem;
            background: #0284c7;
            padding: 3px 8px;
            border-radius: 999px;
            font-weight: 600;
            text-transform: uppercase;
        }
        .health-bar {
            display: flex;
            gap: 16px;
            font-size: 0.85rem;
            color: var(--text-muted);
        }
        .health-item span {
            color: var(--accent);
            font-weight: 600;
        }
        .card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: var(--radius);
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 14px;
        }
        .card-title {
            font-size: 1rem;
            font-weight: 600;
            color: #cbd5e1;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .presets {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }
        .preset-chip {
            background: #334155;
            color: #e2e8f0;
            border: 1px solid #475569;
            padding: 6px 12px;
            border-radius: 999px;
            font-size: 0.8rem;
            cursor: pointer;
            transition: all 0.15s ease;
        }
        .preset-chip:hover {
            background: var(--accent);
            color: #0f172a;
            border-color: var(--accent);
        }
        .form-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }
        @media (max-width: 640px) {
            .form-row { grid-template-columns: 1fr; }
        }
        label {
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--text-muted);
            margin-bottom: 4px;
            display: block;
        }
        textarea, select, input {
            width: 100%;
            background: #0f172a;
            border: 1px solid var(--border-color);
            color: var(--text-main);
            padding: 10px 14px;
            border-radius: 8px;
            font-size: 0.95rem;
            font-family: inherit;
            outline: none;
            transition: border-color 0.2s;
        }
        textarea:focus, select:focus {
            border-color: var(--accent);
        }
        textarea {
            resize: vertical;
            min-height: 80px;
        }
        .btn {
            background: var(--accent);
            color: #0f172a;
            font-weight: 600;
            border: none;
            padding: 10px 18px;
            border-radius: 8px;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            font-size: 0.9rem;
            transition: background 0.15s;
        }
        .btn:hover { background: var(--accent-hover); }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .btn-secondary {
            background: #334155;
            color: #f1f5f9;
        }
        .btn-secondary:hover { background: #475569; }
        .btn-success {
            background: var(--success);
            color: #fff;
        }
        .btn-success:hover { background: #059669; }
        .result-box {
            background: #090d16;
            border: 1px solid #1e293b;
            border-radius: 8px;
            padding: 16px;
            min-height: 70px;
            font-size: 1.25rem;
            line-height: 1.6;
            color: #f8fafc;
            display: flex;
            align-items: center;
        }
        .result-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 14px;
            font-size: 0.8rem;
            color: var(--text-muted);
            margin-top: 6px;
        }
        .meta-pill {
            background: #1e293b;
            padding: 4px 10px;
            border-radius: 6px;
            border: 1px solid #334155;
        }
        .meta-pill strong { color: #f1f5f9; }
        .audio-player-card {
            display: flex;
            flex-direction: column;
            gap: 12px;
            background: #111827;
            border: 1px solid #374151;
            padding: 16px;
            border-radius: 8px;
        }
        .player-controls {
            display: flex;
            align-items: center;
            gap: 10px;
            flex-wrap: wrap;
        }
        .progress-bar-container {
            display: flex;
            align-items: center;
            gap: 12px;
            width: 100%;
        }
        .progress-bar {
            flex: 1;
            -webkit-appearance: none;
            height: 6px;
            border-radius: 3px;
            background: #374151;
            outline: none;
            cursor: pointer;
        }
        .progress-bar::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 14px;
            height: 14px;
            border-radius: 50%;
            background: var(--accent);
            cursor: pointer;
        }
        .time-label {
            font-size: 0.8rem;
            font-variant-numeric: tabular-nums;
            color: var(--text-muted);
            min-width: 75px;
        }
        .status-badge {
            font-size: 0.75rem;
            padding: 3px 8px;
            border-radius: 4px;
            font-weight: 600;
        }
        .status-ready { background: #064e3b; color: #34d399; }
        .status-loading { background: #78350f; color: #fbbf24; }
        .status-error { background: #7f1d1d; color: #f87171; }
        .spinner {
            display: inline-block;
            width: 14px;
            height: 14px;
            border: 2px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 0.8s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header class="header">
            <div>
                <h1>ISL Multilingual Services <span class="badge">v2.0 Neural</span></h1>
                <p style="font-size: 0.85rem; color: var(--text-muted); margin-top: 4px;">
                    Independent Open-Source Machine Translation (IndicTrans2 200M) & Neural TTS (MMS/VITS)
                </p>
            </div>
            <div class="health-bar" id="healthBar">
                <div class="health-item">Status: <span id="healthStatus">Checking...</span></div>
                <div class="health-item">Peak RAM: <span id="healthRam">--</span></div>
            </div>
        </header>

        <!-- Presets -->
        <section class="card">
            <div class="card-title">Quick Test Prompts (Benchmark Sentences)</div>
            <div class="presets">
                <button class="preset-chip" onclick="setPrompt('Hello, how are you?')">1. Hello, how are you?</button>
                <button class="preset-chip" onclick="setPrompt('I need some water.')">2. I need some water.</button>
                <button class="preset-chip" onclick="setPrompt('My name is Arpit.')">3. My name is Arpit.</button>
                <button class="preset-chip" onclick="setPrompt('Can you please help me?')">4. Can you please help me?</button>
                <button class="preset-chip" onclick="setPrompt('I am going to college tomorrow.')">5. I am going to college tomorrow.</button>
                <button class="preset-chip" onclick="setPrompt('Thank you for your support.')">6. Thank you for your support.</button>
            </div>
        </section>

        <!-- Translation Card -->
        <section class="card">
            <div class="card-title">1. Machine Translation Engine</div>
            <div class="form-row">
                <div>
                    <label for="inputText">English Input Sentence</label>
                    <textarea id="inputText" placeholder="Type an English sentence or question...">Hello, how are you?</textarea>
                </div>
                <div>
                    <label for="targetLang">Target Indian Language</label>
                    <select id="targetLang" onchange="onLanguageChange()">
                        <option value="Hindi" selected>Hindi (हिंदी)</option>
                        <option value="Bengali">Bengali (বাংলা)</option>
                        <option value="Marathi">Marathi (मराठी)</option>
                        <option value="Tamil">Tamil (தமிழ்)</option>
                        <option value="Telugu">Telugu (తెలుగు)</option>
                        <option value="Gujarati">Gujarati (ગુજરાતી)</option>
                        <option value="Kannada">Kannada (ಕನ್ನಡ)</option>
                        <option value="Malayalam">Malayalam (മലയാളം)</option>
                        <option value="Punjabi">Punjabi (ਪੰਜਾਬੀ)</option>
                        <option value="Odia">Odia (ଓଡ଼ିଆ)</option>
                    </select>
                    <div style="margin-top: 16px;">
                        <button class="btn" id="btnTranslate" onclick="executeTranslate()">
                            <span id="transSpinner" class="spinner" style="display: none;"></span>
                            Translate to Native Script
                        </button>
                    </div>
                </div>
            </div>

            <div>
                <label>Translated Result (Native Script)</label>
                <div class="result-box" id="translatedBox">--</div>
                <div class="result-meta" id="transMeta" style="display: none;">
                    <span class="meta-pill">Latency: <strong id="transTime">0.0s</strong></span>
                    <span class="meta-pill">Model: <strong id="transModel">--</strong></span>
                    <span class="meta-pill">Cached: <strong id="transCached">No</strong></span>
                </div>
            </div>
        </section>

        <!-- Speech Synthesis Card -->
        <section class="card">
            <div class="card-title">2. Neural Text-to-Speech Engine</div>
            <div class="form-row">
                <div>
                    <label for="voiceSelect">Neural Voice</label>
                    <select id="voiceSelect">
                        <!-- Populated dynamically -->
                    </select>
                </div>
                <div style="display: flex; align-items: flex-end;">
                    <button class="btn btn-success" id="btnSpeak" onclick="executeSpeak()" style="width: 100%;">
                        <span id="speakSpinner" class="spinner" style="display: none;"></span>
                        Synthesize Speech (WAV)
                    </button>
                </div>
            </div>

            <!-- Audio Player -->
            <div class="audio-player-card">
                <div class="player-controls">
                    <button class="btn" id="btnPlay" onclick="audioMgr.play()" disabled>▶ Play</button>
                    <button class="btn btn-secondary" id="btnPause" onclick="audioMgr.pause()" disabled>⏸ Pause</button>
                    <button class="btn btn-secondary" id="btnStop" onclick="audioMgr.stop()" disabled>⏹ Stop</button>
                    <button class="btn btn-secondary" id="btnReplay" onclick="audioMgr.replay()" disabled>↺ Replay</button>
                    <button class="btn btn-secondary" id="btnDownload" onclick="downloadCurrentAudio()" disabled>⤓ Download WAV</button>
                    <span class="status-badge status-ready" id="playerStatus">Ready</span>
                </div>
                <div class="progress-bar-container">
                    <input type="range" class="progress-bar" id="progressBar" min="0" max="100" value="0" oninput="onSeek(this.value)">
                    <div class="time-label" id="timeLabel">0:00 / 0:00</div>
                </div>
                <div class="result-meta" id="ttsMeta" style="display: none;">
                    <span class="meta-pill">Synthesis Time: <strong id="ttsTime">0.0s</strong></span>
                    <span class="meta-pill">Audio Duration: <strong id="ttsDuration">0.0s</strong></span>
                    <span class="meta-pill">Format: <strong>16kHz Mono PCM WAV</strong></span>
                </div>
            </div>
        </section>
    </div>

    <!-- Client Script -->
    <script>
        // Minimal in-page BrowserAudioManager implementation
        class BrowserAudioManager {
            constructor(options = {}) {
                this.audio = new Audio();
                this.currentAudioUrl = null;
                this.currentAudioId = null;
                this.onStateChange = options.onStateChange || (() => {});
                this.onTimeUpdate = options.onTimeUpdate || (() => {});
                this.onError = options.onError || (() => {});

                this.audio.addEventListener("play", () => this.onStateChange("playing"));
                this.audio.addEventListener("pause", () => this.onStateChange("paused"));
                this.audio.addEventListener("ended", () => this.onStateChange("ended"));
                this.audio.addEventListener("timeupdate", () => {
                    const cur = this.audio.currentTime || 0;
                    const dur = this.audio.duration || 0;
                    const pct = dur > 0 ? (cur / dur) * 100 : 0;
                    this.onTimeUpdate(cur, dur, pct);
                });
                this.audio.addEventListener("error", (e) => {
                    const msg = this.audio.error ? this.audio.error.message : "Audio error";
                    this.onError(msg);
                });
            }

            loadSource(url, id = null) {
                this.stop();
                this.currentAudioUrl = url;
                this.currentAudioId = id;
                this.audio.src = url;
                this.audio.load();
            }

            play() { if (this.currentAudioUrl) return this.audio.play(); }
            pause() { this.audio.pause(); }
            stop() { this.audio.pause(); this.audio.currentTime = 0; }
            replay() { this.audio.currentTime = 0; return this.play(); }
            seekToPct(pct) {
                if (this.audio.duration) {
                    this.audio.currentTime = (pct / 100) * this.audio.duration;
                }
            }
        }

        const audioMgr = new BrowserAudioManager({
            onStateChange: (state) => {
                const badge = document.getElementById("playerStatus");
                if (state === "playing") {
                    badge.className = "status-badge status-ready";
                    badge.innerText = "Playing";
                } else if (state === "paused") {
                    badge.innerText = "Paused";
                } else if (state === "ended") {
                    badge.innerText = "Completed";
                }
            },
            onTimeUpdate: (cur, dur, pct) => {
                document.getElementById("progressBar").value = pct;
                document.getElementById("timeLabel").innerText = `${fmtTime(cur)} / ${fmtTime(dur)}`;
            },
            onError: (err) => {
                const badge = document.getElementById("playerStatus");
                badge.className = "status-badge status-error";
                badge.innerText = "Playback Error";
                alert("Audio error: " + err);
            }
        });

        function fmtTime(sec) {
            const m = Math.floor(sec / 60);
            const s = Math.floor(sec % 60);
            return `${m}:${s < 10 ? '0' : ''}${s}`;
        }

        function setPrompt(text) {
            document.getElementById("inputText").value = text;
        }

        function onSeek(val) {
            audioMgr.seekToPct(Number(val));
        }

        let allVoices = [];

        async function fetchHealth() {
            try {
                const res = await fetch("/api/v2/language/health");
                if (res.ok) {
                    const data = await res.json();
                    document.getElementById("healthStatus").innerText = data.status.toUpperCase();
                    document.getElementById("healthRam").innerText = `${data.peak_memory_mb} MB`;
                }
            } catch (e) {
                document.getElementById("healthStatus").innerText = "Offline";
            }
        }

        async function fetchVoices() {
            try {
                const res = await fetch("/api/v2/language/voices");
                if (res.ok) {
                    const data = await res.json();
                    allVoices = data.voices || [];
                    onLanguageChange();
                }
            } catch (e) {
                console.error("Failed to fetch voices:", e);
            }
        }

        function onLanguageChange() {
            const lang = document.getElementById("targetLang").value;
            const select = document.getElementById("voiceSelect");
            select.innerHTML = "";
            const filtered = allVoices.filter(v => v.language_name.toLowerCase() === lang.toLowerCase());
            if (filtered.length === 0) {
                const opt = document.createElement("option");
                opt.value = "";
                opt.innerText = `Default ${lang} Neural Voice`;
                select.appendChild(opt);
            } else {
                filtered.forEach(v => {
                    const opt = document.createElement("option");
                    opt.value = v.voice_id;
                    opt.innerText = `${v.name} (${v.engine.toUpperCase()})`;
                    if (v.is_default) opt.selected = true;
                    select.appendChild(opt);
                });
            }
        }

        async function executeTranslate() {
            const text = document.getElementById("inputText").value.trim();
            const target = document.getElementById("targetLang").value;
            if (!text) { alert("Please enter text to translate"); return; }

            const btn = document.getElementById("btnTranslate");
            const spinner = document.getElementById("transSpinner");
            btn.disabled = true;
            spinner.style.display = "inline-block";

            try {
                const res = await fetch("/api/v2/language/translate", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ text: text, target_lang: target })
                });
                const data = await res.json();
                if (data.success) {
                    document.getElementById("translatedBox").innerText = data.translated_text;
                    document.getElementById("transTime").innerText = `${data.processing_duration}s`;
                    document.getElementById("transModel").innerText = data.model_name;
                    document.getElementById("transCached").innerText = data.cached ? "Yes (Instant)" : "No (Computed)";
                    document.getElementById("transMeta").style.display = "flex";
                } else {
                    document.getElementById("translatedBox").innerText = `Error: ${data.error}`;
                }
            } catch (err) {
                document.getElementById("translatedBox").innerText = `Request error: ${err.message}`;
            } finally {
                btn.disabled = false;
                spinner.style.display = "none";
                fetchHealth();
            }
        }

        async function executeSpeak() {
            const text = document.getElementById("translatedBox").innerText.trim();
            const target = document.getElementById("targetLang").value;
            const voiceId = document.getElementById("voiceSelect").value;

            if (!text || text === "--" || text.startsWith("Error:")) {
                alert("Please translate text first before generating speech!");
                return;
            }

            const btn = document.getElementById("btnSpeak");
            const spinner = document.getElementById("speakSpinner");
            btn.disabled = true;
            spinner.style.display = "inline-block";

            try {
                const res = await fetch("/api/v2/language/speak", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        text: text,
                        target_lang: target,
                        voice_id: voiceId || undefined
                    })
                });
                const data = await res.json();
                if (data.success && data.data_uri) {
                    audioMgr.loadSource(data.data_uri, data.audio_id);
                    document.getElementById("btnPlay").disabled = false;
                    document.getElementById("btnPause").disabled = false;
                    document.getElementById("btnStop").disabled = false;
                    document.getElementById("btnReplay").disabled = false;
                    document.getElementById("btnDownload").disabled = false;

                    document.getElementById("ttsTime").innerText = `${data.processing_duration}s`;
                    document.getElementById("ttsDuration").innerText = `${data.duration_seconds}s`;
                    document.getElementById("ttsMeta").style.display = "flex";
                    audioMgr.play();
                } else {
                    alert("TTS Synthesis Failed: " + (data.error || "Unknown error"));
                }
            } catch (err) {
                alert("Network error: " + err.message);
            } finally {
                btn.disabled = false;
                spinner.style.display = "none";
                fetchHealth();
            }
        }

        function downloadCurrentAudio() {
            if (audioMgr.currentAudioUrl) {
                const filename = `${audioMgr.currentAudioId || "speech"}.wav`;
                const a = document.createElement("a");
                a.href = audioMgr.currentAudioUrl;
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
            }
        }

        // Initialize on load
        window.addEventListener("DOMContentLoaded", () => {
            fetchHealth();
            fetchVoices();
        });
    </script>
</body>
</html>
"""
