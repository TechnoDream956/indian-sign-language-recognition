#!/usr/bin/env bash
# =============================================================================
# ISL Translator — Double-Click Terminal Launcher (macOS)
# =============================================================================
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

echo "=================================================="
echo "  Starting Indian Sign Language (ISL) Translator  "
echo "=================================================="

if [ -f "./venv/bin/python" ]; then
    PYTHON_EXEC="./venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_EXEC="python3"
else
    echo "Error: Python 3 executable not found!"
    exit 1
fi

exec "$PYTHON_EXEC" packaging/launcher.py "$@"
