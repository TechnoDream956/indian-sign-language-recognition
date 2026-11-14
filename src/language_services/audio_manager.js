/**
 * Browser Audio Manager for Multilingual Neural TTS Playback.
 * Provides play, pause, stop, replay, progress tracking, and download capabilities.
 */

class BrowserAudioManager {
    constructor(options = {}) {
        this.audioElement = new Audio();
        this.currentAudioUrl = null;
        this.currentAudioId = null;
        this.isPlaying = false;
        this.isLoading = false;
        
        // Callbacks
        this.onStateChange = options.onStateChange || (() => {});
        this.onTimeUpdate = options.onTimeUpdate || (() => {});
        this.onError = options.onError || (() => {});

        this._setupListeners();
    }

    _setupListeners() {
        this.audioElement.addEventListener("play", () => {
            this.isPlaying = true;
            this.isLoading = false;
            this.onStateChange({ state: "playing" });
        });

        this.audioElement.addEventListener("pause", () => {
            this.isPlaying = false;
            this.onStateChange({ state: "paused" });
        });

        this.audioElement.addEventListener("ended", () => {
            this.isPlaying = false;
            this.onStateChange({ state: "ended" });
        });

        this.audioElement.addEventListener("timeupdate", () => {
            const currentTime = this.audioElement.currentTime || 0;
            const duration = this.audioElement.duration || 0;
            const progress = duration > 0 ? (currentTime / duration) * 100 : 0;
            this.onTimeUpdate({ currentTime, duration, progress });
        });

        this.audioElement.addEventListener("error", (e) => {
            this.isPlaying = false;
            this.isLoading = false;
            const errorMsg = this.audioElement.error ? this.audioElement.error.message : "Audio playback error";
            this.onError(errorMsg);
            this.onStateChange({ state: "error", error: errorMsg });
        });

        this.audioElement.addEventListener("waiting", () => {
            this.isLoading = true;
            this.onStateChange({ state: "buffering" });
        });

        this.audioElement.addEventListener("canplay", () => {
            this.isLoading = false;
            this.onStateChange({ state: "ready" });
        });
    }

    loadSource(audioSource, audioId = null) {
        this.stop();
        this.currentAudioUrl = audioSource;
        this.currentAudioId = audioId;
        this.audioElement.src = audioSource;
        this.audioElement.load();
    }

    play() {
        if (!this.currentAudioUrl) return Promise.reject(new Error("No audio source loaded"));
        return this.audioElement.play().catch(err => {
            this.onError(err.message);
            throw err;
        });
    }

    pause() {
        this.audioElement.pause();
    }

    stop() {
        this.audioElement.pause();
        this.audioElement.currentTime = 0;
        this.isPlaying = false;
        this.onStateChange({ state: "stopped" });
    }

    replay() {
        this.audioElement.currentTime = 0;
        return this.play();
    }

    seek(seconds) {
        if (this.audioElement.duration) {
            this.audioElement.currentTime = Math.max(0, Math.min(seconds, this.audioElement.duration));
        }
    }

    download(filename = "synthesized_speech.wav") {
        if (!this.currentAudioUrl) return;
        const link = document.createElement("a");
        link.href = this.currentAudioUrl;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
}

// Export for module systems or global window
if (typeof module !== "undefined" && module.exports) {
    module.exports = BrowserAudioManager;
} else if (typeof window !== "undefined") {
    window.BrowserAudioManager = BrowserAudioManager;
}
