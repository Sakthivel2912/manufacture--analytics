"""Integration checks for the browser PPE application and real trained model."""

import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import ppe_decision
from src.detector import (
    CONFIDENCE_THRESHOLD,
    MODEL_PATH,
    access_allowed,
    detect,
    evaluate_detections,
    get_device_info,
    load_model,
    read_image,
)
from src.ppe_annotate import draw_decisions


def check_sample(model, relative_image, expected_allowed, device_info):
    image_path = ROOT / relative_image
    image = read_image(image_path)
    if image is None:
        raise AssertionError(f"Sample image could not be read: {image_path}")
    detections, _ = detect(model, image, CONFIDENCE_THRESHOLD, device_info["device"])
    decisions = evaluate_detections(detections, CONFIDENCE_THRESHOLD)
    actual_allowed = access_allowed(decisions)
    if actual_allowed != expected_allowed:
        raise AssertionError(
            f"{image_path.name}: expected {'ALLOWED' if expected_allowed else 'DENIED'}, "
            f"got {'ALLOWED' if actual_allowed else 'DENIED'}; detections={detections}"
        )


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Required trained model does not exist: {MODEL_PATH}")
    print(f"Model path exists: {MODEL_PATH}")

    device_info = get_device_info()
    print(f"CUDA available: {device_info['cuda_available']}")
    print(f"Device: {device_info['label']}")
    print(f"GPU: {device_info['gpu']}")

    model = load_model(device_info["device"])
    print("Model loads: OK")
    print("Detector imports: OK")
    print("PPE decision engine imports: OK")
    print("PPE annotation imports: OK")

    check_sample(
        model,
        Path("dataset") / "images" / "train" / "image1001.jpg",
        True,
        device_info,
    )
    print("image1001 produces ALLOWED: OK")
    check_sample(
        model,
        Path("dataset") / "images" / "train" / "image100.jpg",
        False,
        device_info,
    )
    print("image100 produces DENIED: OK")

    no_person_decisions = evaluate_detections([], CONFIDENCE_THRESHOLD)
    if access_allowed(no_person_decisions) or no_person_decisions:
        raise AssertionError("No-person case must be denied with no person decisions.")
    print("No-person case produces DENIED: OK")

    if read_image(ROOT / "dataset" / "images" / "missing-image.jpg") is not None:
        raise AssertionError("Missing-image case should return None.")
    print("Missing-image case does not crash: OK")

    if not callable(draw_decisions) or not callable(ppe_decision.evaluate_detections):
        raise AssertionError("Existing decision or annotation API is unavailable.")

    test_run = subprocess.run(
        [sys.executable, "-m", "unittest", "tests.test_ppe_decision", "-v"],
        cwd=ROOT,
        check=False,
    )
    if test_run.returncode != 0:
        return test_run.returncode
    print("Existing PPE unit tests pass: OK")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"TEST FAILURE: {exc}", file=sys.stderr)
        sys.exit(1)