from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


@dataclass
class DetectionReport:
    source: str
    prediction: str
    confidence: float
    risk_score: float
    frame_count: int
    details: dict[str, float]


class DeepfakeDetector:
    """Hackathon-friendly deepfake detector using visual artifact signals.

    This baseline intentionally avoids large model weights, so it can run in
    constrained environments. It estimates a fake risk score from:
    - facial edge sharpness consistency (Laplacian variance)
    - color channel imbalance around the face
    - JPEG-like blocking artifacts
    """

    def __init__(self, face_cascade_path: str | None = None) -> None:
        cascade_path = (
            face_cascade_path
            if face_cascade_path
            else cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError(f"Could not load face cascade from {cascade_path}")

    def _iter_video_frames(self, video_path: str | Path, frame_stride: int = 5) -> Iterable[np.ndarray]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % frame_stride == 0:
                yield frame
            idx += 1
        cap.release()

    @staticmethod
    def _jpeg_blocking_score(gray_img: np.ndarray, block_size: int = 8) -> float:
        h, w = gray_img.shape
        if h < block_size or w < block_size:
            return 0.0

        horizontal_edges = np.abs(np.diff(gray_img.astype(np.float32), axis=1))
        vertical_edges = np.abs(np.diff(gray_img.astype(np.float32), axis=0))

        block_cols = list(range(block_size - 1, w - 1, block_size))
        block_rows = list(range(block_size - 1, h - 1, block_size))

        if not block_cols or not block_rows:
            return 0.0

        block_col_mean = float(np.mean(horizontal_edges[:, block_cols]))
        non_block_col_mean = float(np.mean(horizontal_edges))
        block_row_mean = float(np.mean(vertical_edges[block_rows, :]))
        non_block_row_mean = float(np.mean(vertical_edges))

        score = ((block_col_mean - non_block_col_mean) + (block_row_mean - non_block_row_mean)) / 2.0
        return max(0.0, score / 25.0)

    @staticmethod
    def _face_color_mismatch(face_img: np.ndarray) -> float:
        b, g, r = cv2.split(face_img.astype(np.float32))
        rg_diff = abs(float(np.mean(r) - np.mean(g)))
        gb_diff = abs(float(np.mean(g) - np.mean(b)))
        rb_diff = abs(float(np.mean(r) - np.mean(b)))
        return (rg_diff + gb_diff + rb_diff) / (3.0 * 255.0)

    @staticmethod
    def _edge_instability(face_img: np.ndarray) -> float:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return min(1.0, max(0.0, 1.0 - (lap_var / 400.0)))

    def _find_primary_face(self, frame: np.ndarray) -> np.ndarray | None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(48, 48))
        if len(faces) == 0:
            return None
        x, y, w, h = sorted(faces, key=lambda box: box[2] * box[3], reverse=True)[0]
        return frame[y : y + h, x : x + w]

    def analyze_video(self, video_path: str | Path) -> DetectionReport:
        edge_scores: list[float] = []
        color_scores: list[float] = []
        block_scores: list[float] = []
        frame_count = 0

        for frame in self._iter_video_frames(video_path):
            frame_count += 1
            face = self._find_primary_face(frame)
            if face is None:
                continue
            gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            edge_scores.append(self._edge_instability(face))
            color_scores.append(self._face_color_mismatch(face))
            block_scores.append(self._jpeg_blocking_score(gray_face))

        if frame_count == 0:
            raise ValueError("No frames were read from the video")

        if not edge_scores:
            risk_score = 0.5
            details = {
                "edge_instability": 0.5,
                "color_mismatch": 0.5,
                "jpeg_blocking": 0.5,
            }
        else:
            details = {
                "edge_instability": float(np.mean(edge_scores)),
                "color_mismatch": float(np.mean(color_scores)),
                "jpeg_blocking": float(np.mean(block_scores)),
            }
            risk_score = (
                details["edge_instability"] * 0.45
                + details["color_mismatch"] * 0.2
                + details["jpeg_blocking"] * 0.35
            )

        prediction = "FAKE" if risk_score >= 0.5 else "REAL"
        confidence = min(0.99, max(0.51, abs(risk_score - 0.5) * 2.0 + 0.5))

        return DetectionReport(
            source=str(video_path),
            prediction=prediction,
            confidence=confidence,
            risk_score=float(risk_score),
            frame_count=frame_count,
            details=details,
        )

    def analyze_image(self, image_path: str | Path) -> DetectionReport:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Cannot open image: {image_path}")
        face = self._find_primary_face(image)
        if face is None:
            raise ValueError("No face found in image")

        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        details = {
            "edge_instability": self._edge_instability(face),
            "color_mismatch": self._face_color_mismatch(face),
            "jpeg_blocking": self._jpeg_blocking_score(gray_face),
        }
        risk_score = (
            details["edge_instability"] * 0.45
            + details["color_mismatch"] * 0.2
            + details["jpeg_blocking"] * 0.35
        )
        prediction = "FAKE" if risk_score >= 0.5 else "REAL"
        confidence = min(0.99, max(0.51, abs(risk_score - 0.5) * 2.0 + 0.5))

        return DetectionReport(
            source=str(image_path),
            prediction=prediction,
            confidence=float(confidence),
            risk_score=float(risk_score),
            frame_count=1,
            details={k: float(v) for k, v in details.items()},
        )
