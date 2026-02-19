from __future__ import annotations

import time

import streamlit as st

from src.media_pipeline import analyze_media, authenticate_media, detect_media_type

st.set_page_config(page_title="TrustLens", page_icon="🛡️", layout="wide")

if "page" not in st.session_state:
    st.session_state.page = "welcome"
if "uploaded_name" not in st.session_state:
    st.session_state.uploaded_name = None
if "uploaded_bytes" not in st.session_state:
    st.session_state.uploaded_bytes = None
if "media_type" not in st.session_state:
    st.session_state.media_type = None
if "auth_results" not in st.session_state:
    st.session_state.auth_results = None
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None


def _badge(status: str) -> str:
    return "🟢 Trusted" if status == "trusted" else "🟡 Warning"


if st.session_state.page == "welcome":
    st.markdown("<h1 style='text-align:center;'>TrustLens</h1>", unsafe_allow_html=True)
    st.markdown(
        "<h4 style='text-align:center;'>Welcome to TrustLens — Authenticate media and detect deepfakes with explainable AI.</h4>",
        unsafe_allow_html=True,
    )
    st.write("")
    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        if st.button("Get Started", use_container_width=True, type="primary"):
            st.session_state.page = "upload"
            st.rerun()

elif st.session_state.page == "upload":
    st.title("Upload Media")
    st.subheader("Drop your file separately for Audio, Video, and Image")

    col1, col2, col3 = st.columns(3)
    with col1:
        audio_file = st.file_uploader("Audio", type=["wav", "mp3", "m4a", "aac", "flac", "ogg"], key="audio")
    with col2:
        video_file = st.file_uploader("Video", type=["mp4", "mov", "avi", "mkv", "webm"], key="video")
    with col3:
        image_file = st.file_uploader("Image", type=["jpg", "jpeg", "png", "bmp", "webp"], key="image")

    selected = audio_file or video_file or image_file
    if selected:
        st.success(f"Selected file: {selected.name}")

    if st.button("Proceed", type="primary", disabled=selected is None):
        st.session_state.uploaded_name = selected.name
        st.session_state.uploaded_bytes = selected.getvalue()
        st.session_state.media_type = detect_media_type(selected.name)
        st.session_state.page = "auth"
        st.rerun()

elif st.session_state.page == "auth":
    st.title("Authentication Process")
    st.subheader("Verifying file authenticity and integrity")

    st.info(f"Media Type Detected: **{st.session_state.media_type.upper()}** | File: **{st.session_state.uploaded_name}**")

    if st.session_state.auth_results is None:
        with st.spinner("Running authentication methods..."):
            time.sleep(0.7)
            st.session_state.auth_results = authenticate_media(
                st.session_state.uploaded_name,
                st.session_state.uploaded_bytes,
                st.session_state.media_type,
            )

    for item in st.session_state.auth_results:
        with st.container(border=True):
            st.markdown(f"**{item.name}** — {_badge(item.status)}")
            st.caption(item.detail)

    trusted = sum(1 for x in st.session_state.auth_results if x.status == "trusted")
    total = len(st.session_state.auth_results)
    if trusted == total:
        st.success("Authentication successful. File trust checks passed.")
    else:
        st.warning(f"Authentication completed with warnings ({trusted}/{total} trusted).")

    if st.button("Proceed to Deepfake Detection", type="primary"):
        st.session_state.page = "detection"
        st.rerun()

elif st.session_state.page == "detection":
    st.title("Deepfake Detection")

    if st.session_state.analysis_result is None:
        with st.spinner("Running CNN + OpenCV + media-specific AI checks..."):
            st.session_state.analysis_result = analyze_media(
                st.session_state.uploaded_name,
                st.session_state.uploaded_bytes,
                st.session_state.media_type,
            )

    result = st.session_state.analysis_result
    st.subheader("Explained Output")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Deepfake Result", result.deepfake_result)
        st.metric("Risk Score", f"{result.risk_score:.2f}")
        st.metric("Technique Used", result.technique_used)
    with col2:
        st.metric("Media Type", result.media_type.upper())
        st.metric("Frames / Units Analysed", str(result.frames_analyzed))
        st.metric("Unit Sec Taken", f"{result.unit_seconds_taken:.2f}")

    st.markdown("**Explanation**")
    st.write(result.explanation)

    st.markdown("**Steps to Follow**")
    for idx, step in enumerate(result.steps_to_follow, start=1):
        st.write(f"{idx}. {step}")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("New Scan"):
            st.session_state.page = "upload"
            st.session_state.uploaded_name = None
            st.session_state.uploaded_bytes = None
            st.session_state.media_type = None
            st.session_state.auth_results = None
            st.session_state.analysis_result = None
            st.rerun()
    with c2:
        if st.button("Close"):
            st.success("Session closed. You can safely close this tab.")
