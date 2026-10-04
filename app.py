"""Automatic browser camera and image-upload PPE access checker."""

import hashlib
import time

import cv2
import streamlit as st

from src.camera import decode_live_frame, live_camera
from src.detector import (
    CONFIDENCE_THRESHOLD,
    annotate_image,
    detect,
    evaluate_detections,
    get_device_info,
    load_model,
    read_image,
)
from src.ui import show_camera_unavailable, show_decision, show_device


st.set_page_config(page_title="PPE Access Check", page_icon="🦺", layout="wide")


@st.cache_resource
def get_model(device):
    return load_model(device)


def analyze_frame(image, threshold, device_info):
    model = get_model(device_info["device"])
    detections, person_boxes = detect(
        model,
        image,
        confidence_threshold=threshold,
        device=device_info["device"],
    )
    decisions = evaluate_detections(detections, threshold)
    annotated = annotate_image(image, person_boxes, decisions)
    return decisions, cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)


def reset_state():
    st.session_state.input_generation = st.session_state.get("input_generation", 0) + 1
    for key in (
        "upload_analysis_key",
        "upload_analysis",
        "camera_analysis",
        "camera_failure",
        "camera_processed_seq",
        "camera_threshold",
        "camera_last_frame_at",
    ):
        st.session_state.pop(key, None)


@st.fragment(run_every=1.0)
def render_live_camera(threshold, device_info, generation):
    camera_event = live_camera(
        interval_ms=200,
        key=f"live_camera_{generation}",
    )
    if isinstance(camera_event, dict) and camera_event.get("status") == "unavailable":
        st.session_state.pop("camera_analysis", None)
        st.session_state.pop("camera_last_frame_at", None)
        show_camera_unavailable()
        return

    if isinstance(camera_event, dict) and camera_event.get("jpeg"):
        frame_seq = int(camera_event.get("seq", 0))
        is_new_frame = frame_seq != st.session_state.get("camera_processed_seq")
        old_threshold = st.session_state.get("camera_threshold")
        needs_analysis = is_new_frame or old_threshold != float(threshold)
        if needs_analysis:
            if is_new_frame:
                st.session_state.camera_last_frame_at = time.monotonic()
            frame = decode_live_frame(camera_event)
            if frame is None:
                st.session_state.pop("camera_analysis", None)
                st.session_state.camera_failure = (
                    frame_seq,
                    float(threshold),
                    "Camera frame could not be read. Please allow camera access or upload an image.",
                )
                st.session_state.camera_processed_seq = frame_seq
                st.session_state.camera_threshold = float(threshold)
            else:
                try:
                    decisions, annotated = analyze_frame(frame, threshold, device_info)
                    st.session_state.camera_analysis = (frame_seq, float(threshold), decisions, annotated)
                    st.session_state.pop("camera_failure", None)
                    st.session_state.camera_processed_seq = frame_seq
                    st.session_state.camera_threshold = float(threshold)
                except Exception as exc:
                    st.session_state.pop("camera_analysis", None)
                    st.session_state.camera_failure = (
                        frame_seq,
                        float(threshold),
                        f"Inference failed: {exc}",
                    )
                    st.session_state.camera_processed_seq = frame_seq
                    st.session_state.camera_threshold = float(threshold)

        analysis = st.session_state.get("camera_analysis")
        failure = st.session_state.get("camera_failure")
        last_frame_at = st.session_state.get("camera_last_frame_at")
        if last_frame_at is not None and time.monotonic() - last_frame_at > 1.5:
            st.session_state.pop("camera_analysis", None)
            st.session_state.pop("camera_last_frame_at", None)
            show_camera_unavailable()
            return
        if failure is not None and failure[:2] == (frame_seq, float(threshold)):
            st.error("🔴 ACCESS DENIED")
            st.error(failure[2])
        elif analysis is not None and analysis[1] == float(threshold):
            st.subheader("Live analysis")
            st.caption(f"Automatically analyzed camera frame {analysis[0]} (up to 5 frames per second).")
            st.image(analysis[3], caption="Latest analyzed camera frame", width="stretch")
            show_decision(analysis[2])
        else:
            st.error("🔴 ACCESS DENIED")
            st.info("Waiting for the first camera frame...")
        return

    last_frame_at = st.session_state.get("camera_last_frame_at")
    if last_frame_at is not None and time.monotonic() - last_frame_at > 1.0:
        st.session_state.pop("camera_analysis", None)
        st.session_state.pop("camera_last_frame_at", None)
        show_camera_unavailable()
    else:
        st.error("🔴 ACCESS DENIED")
        st.info("Starting camera. Allow browser camera access to begin automatic analysis.")


device_info = get_device_info()
show_device(device_info)
threshold = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.0,
    max_value=1.0,
    value=float(CONFIDENCE_THRESHOLD),
    step=0.05,
)

st.title("PPE Access Check")
st.caption("Live access status based on helmet, vest, and boots.")

if "input_generation" not in st.session_state:
    st.session_state.input_generation = 0

if st.button("Clear / Reset", icon="🔄"):
    reset_state()
    st.rerun()

generation = st.session_state.input_generation
source = st.radio("Image source", ("Upload", "Camera"), horizontal=True)

if source == "Upload":
    uploaded = st.file_uploader(
        "Choose an image",
        type=("jpg", "jpeg", "png", "webp", "bmp"),
        key=f"upload_{generation}",
    )
    if uploaded is None:
        st.info("Upload an image to run the PPE check automatically.")
        st.error("🔴 ACCESS DENIED")
    else:
        image_bytes = uploaded.getvalue()
        image = read_image(image_bytes)
        if image is None:
            st.error("The selected image could not be read. Try a different image file.")
            st.error("🔴 ACCESS DENIED")
        else:
            image_key = (hashlib.sha256(image_bytes).hexdigest(), float(threshold))
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded image", width="stretch")
            if st.session_state.get("upload_analysis_key") != image_key:
                try:
                    with st.spinner("Analyzing image..."):
                        decisions, annotated = analyze_frame(image, threshold, device_info)
                    st.session_state.upload_analysis = (decisions, annotated)
                    st.session_state.upload_analysis_key = image_key
                except Exception as exc:
                    st.error(f"Inference failed: {exc}")
                    st.error("🔴 ACCESS DENIED")
            analysis = st.session_state.get("upload_analysis")
            if st.session_state.get("upload_analysis_key") == image_key and analysis is not None:
                decisions, annotated = analysis
                st.subheader("Analysis result")
                st.image(annotated, caption="Annotated image", width="stretch")
                show_decision(decisions)
else:
    render_live_camera(threshold, device_info, generation)