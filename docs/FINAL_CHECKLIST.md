# Final Manual Tasks Checklist

A prioritized checklist of remaining manual action items for the college presentation and submission team:

1. **Retake Face-Free Live Screenshot**
   - Retake a clear, face-free live desktop screenshot showing the running UI and hands.
   - Save to `docs/report/figures/ui_live_running_desktop.png`.
   - Re-run `python3 scripts/build_report.py` to embed the clean image into the report document and PDF.

2. **Record Multi-Signer Live Benchmark Data**
   - Record live evaluation samples across multiple signers using `tools/collection/record_live_benchmark_samples.py`.
   - Evaluate live top-1 accuracy and report the empirical multi-signer numbers to resolve the pending live validation status.

3. **10-Language Audio Listening Check**
   - Perform human listening checks across all 10 supported regional languages as outlined in `docs/INSTALLATION.md`.
   - Verify speech clarity, voice naturalness, and fallback engine behavior (macOS `say` / pyttsx3 / gTTS).

4. **Verify College AI-Use Declaration Policy**
   - Check the department/college academic integrity policy regarding the AI-use declaration format and permitted tool usage.
   - Update the highlighted `TODO` declaration box in the preliminary pages of the report (`scripts/report_builder/preliminary_pages.py` and regenerated report).
