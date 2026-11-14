#!/usr/bin/env bash
# ==============================================================================
# Indian Sign Language Recognition — College Demonstration Launcher
# ==============================================================================
# Environment: macOS Apple Silicon M2
# Pipeline: MediaPipe Hands (Single & Dual Hand) + Decision Fusion + Indic Translation + TTS
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "===================================================================="
echo "🤟  INDIAN SIGN LANGUAGE (ISL) TRANSLATOR — LIVE SYSTEM"
echo "===================================================================="

# Activate virtual environment
if [ -f "./venv/bin/activate" ]; then
    source ./venv/bin/activate
else
    echo "❌ Error: Virtual environment ./venv not found!"
    exit 1
fi

# Use Port 5001 (Port 5000 is reserved by macOS AirPlay / ControlCenter)
export PORT=${PORT:-5001}
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=2

echo "Starting ISL Recognition Server on: http://127.0.0.1:$PORT"
echo "Press Ctrl+C to stop the application and release webcam."
echo "===================================================================="

./venv/bin/python src/app.py
