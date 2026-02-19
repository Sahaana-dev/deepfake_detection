# Deepfake Detection Hackathon Starter

A lightweight deepfake detection baseline designed for hackathon demos.
It inspects videos or images and computes a **fake risk score** from visual artifacts often seen in manipulated media.

## What this project does

- Detects the primary face in each sampled video frame.
- Computes three interpretable signals:
  - **Edge instability**: unnaturally smooth or inconsistent facial details.
  - **Color mismatch**: odd channel balance that can appear after face swaps.
  - **JPEG blocking artifacts**: compression patterns amplified by generation/editing pipelines.
- Produces:
  - `prediction`: `REAL` or `FAKE`
  - `confidence`: confidence score (0.51 to 0.99)
  - per-signal diagnostics for demo explainability.

> This is a practical baseline, not SOTA. For production, add a learned temporal model (e.g., X3D/ViT + face tracking) and evaluate on public deepfake benchmarks.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Analyze a video

```bash
PYTHONPATH=src python src/cli.py path/to/video.mp4 --as-json
```

### Analyze an image

```bash
PYTHONPATH=src python src/cli.py path/to/image.jpg
```

## Hackathon upgrade ideas

1. Add model inference (EfficientNet / TimeSformer) and blend with these heuristic signals.
2. Add Streamlit UI for drag-and-drop uploads with frame-by-frame heatmaps.
3. Save suspicious frame thumbnails to a report folder.
4. Add benchmark script with AUC / F1 against DFDC or FaceForensics++ metadata.

## Project layout

- `src/deepfake_detector.py`: core detection logic.
- `src/cli.py`: command-line entrypoint.
- `tests/test_detector.py`: unit tests for artifact signals.
