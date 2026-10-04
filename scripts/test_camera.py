"""Simple webcam test (no PPE detection yet). Press Q to quit.

Usage (from the project root, with .venv activated):
    python scripts/test_camera.py
    python scripts/test_camera.py --camera 1   # try another camera
"""

import argparse
import sys


def open_camera(index):
    import cv2

    # On Windows, DirectShow usually opens webcams faster and more reliably.
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(index)


def main():
    parser = argparse.ArgumentParser(description="Test webcam access with OpenCV.")
    parser.add_argument("--camera", type=int, default=0, help="camera index (0 = default webcam)")
    args = parser.parse_args()

    try:
        import cv2
    except ImportError:
        print("OpenCV is not installed. Activate .venv and run: pip install opencv-python")
        return 1

    cap = open_camera(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}.")
        print("Check that a webcam is connected, not used by another app (Zoom/Teams),")
        print("and that camera access is allowed in Windows Settings > Privacy > Camera.")
        return 1

    print("Camera opened. Press Q in the video window to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read a frame from the camera.")
                return 1
            cv2.putText(frame, "Camera test - press Q to quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Camera Test", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
    except cv2.error as e:
        print(f"OpenCV display error: {e}")
        return 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print("Camera test finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
