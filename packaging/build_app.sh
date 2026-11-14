#!/usr/bin/env bash
# =============================================================================
# ISL Translator — macOS .app Build Script
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_BIN="$PROJECT_ROOT/venv/bin"

echo "=============================================="
echo "  ISL Translator — macOS .app Builder"
echo "=============================================="
echo "  Project root: $PROJECT_ROOT"
echo "  Python:       $($VENV_BIN/python --version)"
echo "  PyInstaller:  $($VENV_BIN/pyinstaller --version)"
echo "  Architecture: $(arch)"
echo "=============================================="

BUILD_START=$(date +%s)

# ---------------------------------------------------------------------------
#  Step 1: Pre-build model hash verification
# ---------------------------------------------------------------------------
echo ""
echo "[1/4] Verifying source model hashes..."

verify_hash() {
    local FILE="$1"
    local EXPECTED="$2"
    local FULL_PATH="$PROJECT_ROOT/$FILE"
    local ACTUAL
    ACTUAL=$(shasum -a 256 "$FULL_PATH" | awk '{print $1}')
    if [ "$ACTUAL" = "$EXPECTED" ]; then
        echo "  ✓ $FILE"
        return 0
    else
        echo "  ✗ $FILE HASH MISMATCH!"
        echo "    Expected: $EXPECTED"
        echo "    Actual:   $ACTUAL"
        return 1
    fi
}

ALL_PASS=true
verify_hash "models/alphabet_model_26class.h5"  "af429c007ba6e62b20385ac6e194898209c3c5cc44f781fc691f34bcb4f295b3" || ALL_PASS=false
verify_hash "models/alphabet_label_map.json"    "5bd5f98b105afa08aac4fb98efec50d763afc145611938b459f9a87afbe3f024" || ALL_PASS=false

if [ "$ALL_PASS" = false ]; then
    echo "FATAL: Source model hashes do not match. Aborting build."
    exit 1
fi
echo "  All production alphabet model hashes verified."

# ---------------------------------------------------------------------------
#  Step 2: Run PyInstaller
# ---------------------------------------------------------------------------
echo ""
echo "[2/4] Building .app bundle with PyInstaller..."

cd "$PROJECT_ROOT"
"$VENV_BIN/pyinstaller" packaging/ISL_Translator.spec \
    --noconfirm \
    --clean \
    --distpath "$PROJECT_ROOT/dist" \
    --workpath "$PROJECT_ROOT/build"

echo "  Build complete."

# ---------------------------------------------------------------------------
#  Step 3: Post-build model hash verification
# ---------------------------------------------------------------------------
echo ""
echo "[3/4] Verifying bundled model hashes..."

# Find where models ended up
BUNDLE_MODELS=""
if [ -d "$PROJECT_ROOT/dist/ISL Translator.app/Contents/Resources/models" ]; then
    BUNDLE_MODELS="$PROJECT_ROOT/dist/ISL Translator.app/Contents/Resources/models"
elif [ -d "$PROJECT_ROOT/dist/ISL Translator/models" ]; then
    BUNDLE_MODELS="$PROJECT_ROOT/dist/ISL Translator/models"
else
    echo "  Searching for bundled models..."
    MODEL_LOCATION=$(find "$PROJECT_ROOT/dist" -name "alphabet_model_26class.h5" -type f 2>/dev/null | head -1)
    if [ -n "$MODEL_LOCATION" ]; then
        BUNDLE_MODELS=$(dirname "$MODEL_LOCATION")
        echo "  Found at: $BUNDLE_MODELS"
    fi
fi

if [ -z "$BUNDLE_MODELS" ] || [ ! -d "$BUNDLE_MODELS" ]; then
    echo "  WARNING: Could not locate bundled models directory."
    echo "  Manual verification needed."
else
    verify_bundle_hash() {
        local BASENAME="$1"
        local EXPECTED="$2"
        local BUNDLE_FILE="$BUNDLE_MODELS/$BASENAME"
        if [ ! -f "$BUNDLE_FILE" ]; then
            echo "  ✗ $BASENAME NOT FOUND in bundle!"
            return 1
        fi
        local ACTUAL
        ACTUAL=$(shasum -a 256 "$BUNDLE_FILE" | awk '{print $1}')
        if [ "$ACTUAL" = "$EXPECTED" ]; then
            echo "  ✓ $BASENAME"
            return 0
        else
            echo "  ✗ $BASENAME HASH MISMATCH!"
            return 1
        fi
    }

    ALL_PASS=true
    verify_bundle_hash "alphabet_model_26class.h5"  "af429c007ba6e62b20385ac6e194898209c3c5cc44f781fc691f34bcb4f295b3" || ALL_PASS=false
    verify_bundle_hash "alphabet_label_map.json"    "5bd5f98b105afa08aac4fb98efec50d763afc145611938b459f9a87afbe3f024" || ALL_PASS=false

    if [ "$ALL_PASS" = false ]; then
        echo "FATAL: Bundled model hashes do not match. Build is CORRUPT."
        exit 1
    fi
    echo "  All production alphabet bundled model hashes verified."
fi

# ---------------------------------------------------------------------------
#  Step 4: Report
# ---------------------------------------------------------------------------
BUILD_END=$(date +%s)
BUILD_DURATION=$((BUILD_END - BUILD_START))

echo ""
echo "=============================================="
echo "  BUILD REPORT"
echo "=============================================="
echo "  Duration:     ${BUILD_DURATION}s"

APP_PATH="$PROJECT_ROOT/dist/ISL Translator.app"
if [ -d "$APP_PATH" ]; then
    APP_SIZE=$(du -sh "$APP_PATH" | awk '{print $1}')
    echo "  App path:     $APP_PATH"
    echo "  App size:     $APP_SIZE"
else
    echo "  WARNING: .app bundle not found at expected location."
fi

echo "  Architecture: $(arch)"
echo "  Python:       $($VENV_BIN/python --version 2>&1)"
echo "  PyInstaller:  $($VENV_BIN/pyinstaller --version)"
echo "=============================================="
echo "  BUILD SUCCESS"
echo "=============================================="
