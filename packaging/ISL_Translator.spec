# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for ISL Translator macOS .app bundle.

Build command:
    cd /path/to/Indian-Sign-Language-Recognition
    ./venv/bin/pyinstaller packaging/ISL_Translator.spec --noconfirm

Output:
    dist/ISL Translator.app
"""

import os
import sys
from pathlib import Path

# Project root (one level up from this spec file's directory)
# SPECPATH is set by PyInstaller to the directory containing this .spec file
PROJECT_ROOT = os.path.dirname(SPECPATH)
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

# ---------------------------------------------------------------------------
#  Data files to bundle
# ---------------------------------------------------------------------------

# Models — only the production alphabet model and label map
model_datas = [
    (os.path.join(MODELS_DIR, 'alphabet_model_26class.h5'), 'models'),
    (os.path.join(MODELS_DIR, 'alphabet_label_map.json'), 'models'),
]

# Templates and static assets
web_datas = [
    (os.path.join(SRC_DIR, 'templates'), os.path.join('src', 'templates')),
    (os.path.join(SRC_DIR, 'static'), os.path.join('src', 'static')),
]

# Source modules (collected as data so they can be imported by launcher)
src_modules = [
    (os.path.join(SRC_DIR, 'app.py'), 'src'),
    (os.path.join(SRC_DIR, 'inference_engine.py'), 'src'),
    (os.path.join(SRC_DIR, 'alphabet_engine.py'), 'src'),
    (os.path.join(SRC_DIR, 'mediapipe_utils.py'), 'src'),
    (os.path.join(SRC_DIR, 'translator.py'), 'src'),
    (os.path.join(SRC_DIR, 'tts_engine.py'), 'src'),
]

# MediaPipe runtime data files (.binarypb graph definitions, .tflite models)
# These are loaded by file path at runtime, not via Python imports.
import site
_sp = os.path.join(PROJECT_ROOT, 'venv', 'lib', 'python3.10', 'site-packages')
mediapipe_datas = [
    (os.path.join(_sp, 'mediapipe', 'modules'), os.path.join('mediapipe', 'modules')),
]

all_datas = model_datas + web_datas + src_modules + mediapipe_datas

# ---------------------------------------------------------------------------
#  Analysis
# ---------------------------------------------------------------------------

a = Analysis(
    [os.path.join(PROJECT_ROOT, 'packaging', 'launcher.py')],
    pathex=[SRC_DIR, PROJECT_ROOT],
    binaries=[],
    datas=all_datas,
    hiddenimports=[
        # TensorFlow / Keras
        'tf_keras',
        'tf_keras.models',
        'tf_keras.layers',
        'tf_keras.optimizers',
        'tf_keras.src',
        'tf_keras.src.engine',
        'tf_keras.src.engine.base_layer',
        'tf_keras.src.engine.base_layer_v1',
        'tf_keras.src.engine.functional',
        'tf_keras.src.engine.sequential',
        'tf_keras.src.engine.training',
        'tf_keras.src.engine.input_layer',
        'tf_keras.src.layers',
        'tf_keras.src.optimizers',
        'tf_keras.src.saving',
        'tf_keras.src.saving.legacy',
        'tf_keras.src.saving.legacy.hdf5_format',
        'tf_keras.src.utils',
        'tensorflow',
        'tensorflow.python',
        'tensorflow.python.eager',
        'h5py',
        # MediaPipe
        'mediapipe',
        'mediapipe.python',
        'mediapipe.python.solutions',
        'mediapipe.python.solutions.holistic',
        'mediapipe.python.solutions.hands',
        'mediapipe.python.solutions.drawing_utils',
        'mediapipe.python.solutions.drawing_styles',
        # OpenCV
        'cv2',
        # scikit-learn (for pickle loading LabelEncoder)
        'sklearn',
        'sklearn.preprocessing',
        'sklearn.preprocessing._label',
        # Flask
        'flask',
        'flask.json',
        'jinja2',
        'werkzeug',
        'markupsafe',
        # NumPy
        'numpy',
        # PIL
        'PIL',
        'PIL.Image',
        'PIL.ImageDraw',
        'PIL.ImageFont',
        # Standard library
        'pickle',
        'json',
        'threading',
        'subprocess',
        'webbrowser',
        'socket',
        'signal',
        'collections',
        # Project modules
        'app',
        'inference_engine',
        'alphabet_engine',
        'mediapipe_utils',
        'translator',
        'tts_engine',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # NOTE: matplotlib is NOT excluded because mediapipe.drawing_utils imports it
        'pandas',
        'seaborn',
        'IPython',
        'jupyter',
        'notebook',
        'pytest',
        'tkinter',
        '_tkinter',
        'PyQt5',
        'PyQt6',
        'PySide2',
        'PySide6',
        'wx',
        'pygments',
        'docutils',
        'sphinx',
        'tensorboard',
        'tensorboard_data_server',
        'google.cloud',
        'grpc',
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

# ---------------------------------------------------------------------------
#  EXE + BUNDLE
# ---------------------------------------------------------------------------

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ISL Translator',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # Windowed mode — no terminal window
    target_arch='arm64',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name='ISL Translator',
)

app = BUNDLE(
    coll,
    name='ISL Translator.app',
    icon=None,  # No custom icon yet
    bundle_identifier='com.isltranslator.app',
    version='1.0.0',
    info_plist={
        'CFBundleName': 'ISL Translator',
        'CFBundleDisplayName': 'ISL Translator',
        'CFBundleShortVersionString': '1.0.0',
        'CFBundleVersion': '1.0.0',
        'CFBundleIdentifier': 'com.isltranslator.app',
        'CFBundlePackageType': 'APPL',
        'NSCameraUsageDescription': 'ISL Translator requires camera access to recognize Indian Sign Language gestures.',
        'NSHighResolutionCapable': True,
        'LSMinimumSystemVersion': '12.0',
        'LSUIElement': False,
    },
)
