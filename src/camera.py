"""Browser camera component and frame decoding helpers."""

import base64
from pathlib import Path

import streamlit.components.v1 as components

from src.detector import read_image


_COMPONENT_DIR = Path(__file__).resolve().parent / "camera_component"
_live_camera_component = components.declare_component(
    "ppe_live_camera_v4",
    path=str(_COMPONENT_DIR),
)


def live_camera(interval_ms=200, key=None):
    """Render the browser camera and stream rate-limited frames to Python."""
    return _live_camera_component(
        interval_ms=int(interval_ms),
        key=key,
        default=None,
    )


def decode_live_frame(camera_event):
    """Decode the JPEG frame from a live-camera component event."""
    if not isinstance(camera_event, dict) or not camera_event.get("jpeg"):
        return None
    try:
        encoded = base64.b64decode(camera_event["jpeg"], validate=True)
    except (ValueError, TypeError):
        return None
    return read_image(encoded)