from src.media_pipeline import authenticate_media, detect_media_type


def test_detect_media_type_routes_extensions():
    assert detect_media_type("a.mp4") == "video"
    assert detect_media_type("a.wav") == "audio"
    assert detect_media_type("a.jpg") == "image"


def test_authentication_includes_media_specific_checks_for_audio():
    checks = authenticate_media("voice.wav", b"RIFF" + b"1" * 8000, "audio")
    names = [c.name for c in checks]
    assert "Audio Signature Verification" in names
    assert "Spectral Consistency Check" in names
    assert "EXIF Data Analysis" not in names


def test_authentication_includes_media_specific_checks_for_image():
    checks = authenticate_media("face.jpg", b"Exif" + b"2" * 9000, "image")
    names = [c.name for c in checks]
    assert "EXIF Data Analysis" in names
    assert "Frame Integrity Check" not in names
