import numpy as np

from deepfake_detector import DeepfakeDetector


def test_jpeg_blocking_score_non_negative():
    img = np.random.randint(0, 255, size=(64, 64), dtype=np.uint8)
    score = DeepfakeDetector._jpeg_blocking_score(img)
    assert score >= 0.0


def test_face_color_mismatch_range():
    face = np.zeros((32, 32, 3), dtype=np.uint8)
    face[..., 2] = 255
    score = DeepfakeDetector._face_color_mismatch(face)
    assert 0.0 <= score <= 1.0


def test_edge_instability_range():
    face = np.random.randint(0, 255, size=(64, 64, 3), dtype=np.uint8)
    score = DeepfakeDetector._edge_instability(face)
    assert 0.0 <= score <= 1.0
