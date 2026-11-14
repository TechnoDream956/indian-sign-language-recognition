# ISL Translator — macOS Packaging

## Prerequisites

- macOS 12+ (Monterey or later)
- Apple Silicon (ARM64) Mac
- Python 3.10 virtualenv at `../venv/`
- PyInstaller (`pip install pyinstaller`)

## Build

```bash
cd /path/to/Indian-Sign-Language-Recognition
bash packaging/build_app.sh
```

The script will:
1. Verify all 6 model SHA-256 hashes
2. Build the `.app` bundle via PyInstaller
3. Verify bundled model hashes
4. Report build size and duration

## Output

```
dist/ISL Translator.app
```

## Running

Double-click `ISL Translator.app` or:

```bash
open "dist/ISL Translator.app"
```

The application will:
- Start the Flask server on `127.0.0.1:5001`
- Open your default browser automatically
- Begin camera capture and ML inference

## Distribution

```bash
cd dist
zip -r ISL_Translator_macOS.zip "ISL Translator.app"
```

## Known Limitations

- **Unsigned**: macOS Gatekeeper may warn on first launch. Right-click → Open to bypass.
- **Camera Permission**: macOS will prompt for camera access on first launch.
- **TTS**: Uses macOS native `/usr/bin/say`. Indic language voices must be installed via System Preferences → Accessibility → Spoken Content.
- **Architecture**: Built for ARM64 (Apple Silicon). Intel Macs require a separate build.

## Files

| File | Purpose |
|------|---------|
| `launcher.py` | Production entry point for the `.app` |
| `ISL_Translator.spec` | PyInstaller build specification |
| `build_app.sh` | Automated build + verification script |
| `README.md` | This file |
