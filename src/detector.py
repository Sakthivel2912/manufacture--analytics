"""YOLO loading and adaptation for the existing PPE decision engine."""

from pathlib import Path
from threading import RLock

import cv2
import numpy as np
import torch

from src import ppe_decision


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "runs" / "detect" / "ppe_training" / "weights" / "best.pt"
CONFIDENCE_THRESHOLD = ppe_decision.CONFIDENCE_THRESHOLD
CLASS_NAMES = {
    0: "helmet",
    1: "gloves",
    2: "vest",
    3: "boots",
    4: "goggles",
    6: "Person",
}
_DECISION_LOCK = RLock()


def get_device_info():
    """Return the active inference device and its display name."""
    if torch.cuda.is_available():
        return {
            "device": "cuda:0",
            "label": "CUDA",
            "gpu": torch.cuda.get_device_name(0),
            "cuda_available": True,
        }
    return {
        "device": "cpu",
        "label": "CPU",
        "gpu": "Not available",
        "cuda_available": False,
    }


def load_model(device=None):
    """Load the project's trained model without downloading or training."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Trained PPE model not found: {MODEL_PATH}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is unavailable. Activate the project Python environment.") from exc

    try:
        return YOLO(str(MODEL_PATH))
    except Exception as exc:
        raise RuntimeError(f"Could not load the trained PPE model at {MODEL_PATH}: {exc}") from exc


def read_image(image_source):
    """Decode a path, byte string, or Streamlit uploaded image to OpenCV BGR."""
    if isinstance(image_source, (str, Path)):
        image = cv2.imread(str(image_source))
        return image

    if hasattr(image_source, "getvalue"):
        image_source = image_source.getvalue()
    if hasattr(image_source, "read"):
        image_source = image_source.read()
    if image_source is None:
        return None

    try:
        encoded = np.frombuffer(image_source, dtype=np.uint8)
        if encoded.size == 0:
            return None
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    except (TypeError, ValueError, cv2.error):
        return None


def detect(model, image, confidence_threshold=CONFIDENCE_THRESHOLD, device=None):
    """Return normalized detection dictionaries and their person boxes."""
    results = model.predict(
        source=image,
        conf=float(confidence_threshold),
        device=device,
        verbose=False,
    )
    if not results:
        return [], []

    detections = []
    person_boxes = []
    for box in results[0].boxes:
        class_id = int(box.cls[0].item())
        name = CLASS_NAMES.get(class_id)
        if name is None:
            continue
        coordinates = [float(value) for value in box.xyxy[0].tolist()]
        detection = {
            "name": name,
            "class_id": class_id,
            "confidence": float(box.conf[0].item()),
            "box": coordinates,
        }
        detections.append(detection)
        if name == "Person":
            person_boxes.append(coordinates)
    return detections, person_boxes


def evaluate_detections(detections, confidence_threshold=CONFIDENCE_THRESHOLD):
    """Evaluate detections with the existing engine at the selected threshold."""
    threshold = float(confidence_threshold)
    with _DECISION_LOCK:
        original_threshold = ppe_decision.CONFIDENCE_THRESHOLD
        try:
            ppe_decision.CONFIDENCE_THRESHOLD = threshold
            return ppe_decision.evaluate_detections(detections)
        finally:
            ppe_decision.CONFIDENCE_THRESHOLD = original_threshold


def access_allowed(decisions):
    """Return false for no-person frames and require every person to pass."""
    return bool(decisions) and all(decision.allowed for decision in decisions)


def annotate_image(image, person_boxes, decisions):
    """Draw the existing engine's per-person annotation on an image copy."""
    from src.ppe_annotate import draw_decisions

    return draw_decisions(image.copy(), person_boxes, decisions)