# TrustLens — Media Authentication + Deepfake Detection

TrustLens is a hackathon-ready web app with a multi-step workflow:

1. **Welcome page** with project branding and **Get Started**.
2. **Upload page** with separate horizontal upload inputs for **audio, video, image**.
3. **Authentication Process** page with media-specific checks (watermark, provenance, cryptographic hash + type-specific methods).
4. **Deepfake Detection** page using OpenCV + CNN-style feature extraction + media-specific analysis.
5. **Explained output** with deepfake result, risk score, techniques used, frames/units analyzed, runtime, and next steps.

## Run locally (no VS Code required)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown by Streamlit in your browser.

## Key files

- `app.py`: full multi-page UI workflow.
- `src/media_pipeline.py`: media type routing, authentication checks, deepfake analysis.
- `src/deepfake_detector.py`: OpenCV-based image/video deepfake artifact analyzer.
