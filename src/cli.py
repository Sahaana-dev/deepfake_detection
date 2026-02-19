from __future__ import annotations

import json
from pathlib import Path

import click

from deepfake_detector import DeepfakeDetector


@click.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option("--as-json", "as_json", is_flag=True, help="Print report as JSON")
def main(input_path: Path, as_json: bool) -> None:
    """Run deepfake detection against a video or image path."""
    detector = DeepfakeDetector()
    suffix = input_path.suffix.lower()

    if suffix in {".mp4", ".mov", ".avi", ".mkv"}:
        report = detector.analyze_video(input_path)
    else:
        report = detector.analyze_image(input_path)

    if as_json:
        click.echo(json.dumps(report.__dict__, indent=2))
        return

    click.echo(f"Source: {report.source}")
    click.echo(f"Prediction: {report.prediction}")
    click.echo(f"Confidence: {report.confidence:.2f}")
    click.echo(f"Risk score: {report.risk_score:.2f}")
    click.echo(f"Processed frames: {report.frame_count}")
    click.echo("Signals:")
    for name, score in report.details.items():
        click.echo(f"  - {name}: {score:.3f}")


if __name__ == "__main__":
    main()
