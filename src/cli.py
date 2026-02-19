from __future__ import annotations

import argparse
import json
from pathlib import Path



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run deepfake detection against an image or video."
    )
    parser.add_argument("input_path", type=Path, help="Path to an image or video file")
    parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Print output as JSON",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.input_path.exists():
        parser.error(f"File not found: {args.input_path}")

    from deepfake_detector import DeepfakeDetector

    detector = DeepfakeDetector()
    suffix = args.input_path.suffix.lower()

    if suffix in {".mp4", ".mov", ".avi", ".mkv"}:
        report = detector.analyze_video(args.input_path)
    else:
        report = detector.analyze_image(args.input_path)

    if args.as_json:
        print(json.dumps(report.__dict__, indent=2))
        return

    print(f"Source: {report.source}")
    print(f"Prediction: {report.prediction}")
    print(f"Confidence: {report.confidence:.2f}")
    print(f"Risk score: {report.risk_score:.2f}")
    print(f"Processed frames: {report.frame_count}")
    print("Signals:")
    for name, score in report.details.items():
        print(f"  - {name}: {score:.3f}")


if __name__ == "__main__":
    main()
