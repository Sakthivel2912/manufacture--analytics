"""Print the Python / PyTorch / YOLO environment and check it is ready.

Usage (from the project root, with .venv activated):
    python scripts/check_environment.py
"""

import importlib
import platform
import sys


def version_of(module_name, attr="__version__"):
    try:
        module = importlib.import_module(module_name)
        return getattr(module, attr, "unknown"), None
    except Exception as e:
        return None, e


def main():
    problems = []

    print(f"Python:\n  {platform.python_version()}")
    print(f"Python executable:\n  {sys.executable}")
    in_venv = sys.prefix != sys.base_prefix
    print(f"Virtual environment active:\n  {'Yes (' + sys.prefix + ')' if in_venv else 'NO'}")
    if not in_venv:
        problems.append("Not running inside .venv. Activate it first (see README).")
    if sys.version_info[:2] not in ((3, 10), (3, 11), (3, 12)):
        problems.append(f"Python {platform.python_version()} is not 3.10/3.11/3.12 (recommended for YOLO).")

    torch_ver, err = version_of("torch")
    print(f"PyTorch:\n  {torch_ver or 'NOT INSTALLED'}")
    if err:
        problems.append(f"PyTorch import failed: {err}")
    else:
        import torch

        cuda = torch.cuda.is_available()
        print(f"CUDA available:\n  {cuda}")
        if cuda:
            print(f"GPU:\n  {torch.cuda.get_device_name(0)}")
        else:
            print("  (training will run on CPU - slower but works)")

    for label, mod in (("Torchvision", "torchvision"), ("Ultralytics", "ultralytics"),
                       ("OpenCV", "cv2"), ("NumPy", "numpy")):
        ver, err = version_of(mod)
        print(f"{label}:\n  {ver or 'NOT INSTALLED'}")
        if err:
            problems.append(f"{label} import failed: {err}")

    if not problems:
        try:
            from ultralytics import YOLO  # noqa: F401
        except Exception as e:
            problems.append(f"'from ultralytics import YOLO' failed: {e}")

    print("\nEnvironment:")
    if problems:
        print("  X NOT READY")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("  \u2713 READY" if (sys.stdout.encoding or "").lower().startswith("utf") else "  READY")
    return 0


if __name__ == "__main__":
    sys.exit(main())
