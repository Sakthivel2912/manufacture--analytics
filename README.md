# AI-Based Worker Safety and PPE Compliance Detection System

Manufacturing Analytics course project. This stage covers **only** the Python / YOLO
development environment, the PPE dataset, dataset validation, the training script,
model testing and a webcam test. (No web UI, Flask, database or dashboard yet. No face
recognition and no worker identity data.)

---

## 1. Project overview

A YOLO object-detection model is trained to recognise workers and their PPE:

| Dataset class | Shown in the app as        |
|---------------|----------------------------|
| `Person`      | Worker                     |
| `helmet`      | Safety Helmet              |
| `gloves`      | Safety Gloves              |
| `vest`        | Safety Jacket / Safety Vest|
| `boots`       | Safety Shoes               |
| `goggles`     | Safety Glasses             |

```
AI-PPE-Safety-System/
├── .venv/                  # Python virtual environment (created by you, not committed)
├── dataset/
│   ├── data.yaml           # dataset config (classes + paths)
│   ├── images/{train,val,test}/
│   └── labels/{train,val,test}/
├── models/                 # pretrained YOLO weights are downloaded here (yolo26n.pt)
├── training/train.py       # training script
├── scripts/
│   ├── check_environment.py
│   ├── download_dataset.py
│   ├── check_dataset.py
│   ├── test_model.py
│   └── test_camera.py
├── tests/
├── requirements.txt
├── README.md
└── .gitignore
```

## 2. System requirements

- Windows 10/11 (also works on Linux/macOS)
- **Python 3.10 or 3.11** (3.12 also works). Download from https://www.python.org/downloads/
  and tick **"Add python.exe to PATH"** during installation.
- 8 GB RAM recommended, ~3 GB free disk (packages + dataset)
- A webcam (only for the camera test)
- NVIDIA GPU is **optional**. Without one, training runs on the CPU (slower but works).

## 3. Python version

Check before doing anything else:

```cmd
python --version
```

If it shows 3.10.x or 3.11.x you are fine. If it shows 3.13+ or 3.9 or older, install
Python 3.11 alongside it (do not uninstall your current Python) and create the venv with
`py -3.11 -m venv .venv` instead of `python -m venv .venv`.

## 4. Create the virtual environment

Open a terminal **in the `AI-PPE-Safety-System` folder**.

**Windows CMD:**
```cmd
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

**Windows PowerShell:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell says *"running scripts is disabled on this system"*, run this once and
then activate again:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## 5. Activate the virtual environment (every time you open a new terminal)

- CMD: `.venv\Scripts\activate`
- PowerShell: `.\.venv\Scripts\Activate.ps1`

Your prompt then starts with `(.venv)`. Check that Python comes from the venv:

```cmd
where python
python --version
where pip
```
The **first** line of `where python` must be
`...\AI-PPE-Safety-System\.venv\Scripts\python.exe`.
(In PowerShell use `where.exe python` or `Get-Command python`.)

To leave the venv: `deactivate`.

## 6. Install packages

> **Install packages ONLY after activating `.venv`** (prompt shows `(.venv)`).
> Never install them into your system Python.

```cmd
pip install -r requirements.txt
```

This installs Ultralytics, PyTorch (CPU build), torchvision, OpenCV, NumPy, Pillow,
Matplotlib, PyYAML and pandas.

### NVIDIA GPU (optional)
First check if you have one: run `nvidia-smi`. If the command is not found, you have no
NVIDIA GPU (or no driver) - **skip this**, the CPU build is correct.
If `nvidia-smi` works, replace the CPU PyTorch with the CUDA build (inside `.venv`):
```cmd
pip uninstall -y torch torchvision
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
```
Then `python -c "import torch; print(torch.cuda.is_available())"` should print `True`.

## 7. Check the environment

```cmd
python scripts\check_environment.py
yolo checks
python -c "from ultralytics import YOLO; print('YOLO is working')"
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

`check_environment.py` must end with `Environment: ✓ READY`.

### Verified reference environment
Setup was verified on a reference machine (CPU only, no NVIDIA GPU):

| Item | Value |
|------|-------|
| Python | 3.10.12 (inside `.venv`) |
| PyTorch | 2.14.1+cpu |
| CUDA available | False |
| torchvision | 0.29.1+cpu |
| Ultralytics | 8.4.173 |
| OpenCV | 5.0.0 |
| NumPy | 2.2.6 |
| `yolo checks` | Setup complete ✅ |

Re-run the command above on **your** laptop and note your own values here:
- PyTorch version: ______  CUDA available: ______

## 8. Dataset setup

We use the official, pre-annotated **Ultralytics Construction-PPE** dataset
(https://docs.ultralytics.com/datasets/detect/construction-ppe/, AGPL-3.0, ~178 MB,
YOLO format). Download it with:

```cmd
python scripts\download_dataset.py
```

This downloads the official zip from
`https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip`
and extracts it into `dataset/` (keeping our `dataset/data.yaml`).

Manual alternative: download the zip from the URL above and extract it so that the
`images` and `labels` folders end up directly inside `dataset/`.

## 9. Dataset structure

```
dataset/
├── data.yaml
├── images/
│   ├── train/   1132 images
│   ├── val/      143 images
│   └── test/     141 images
└── labels/
    ├── train/   one .txt per image
    ├── val/
    └── test/
```

Each label `.txt` has one line per object: `class_id x_center y_center width height`
(values normalised 0-1).

**Classes** (official order - must not be changed, the label files use these IDs):

| ID | Class | | ID | Class |
|----|-------|-|----|-------|
| 0 | helmet  | | 6 | Person |
| 1 | gloves  | | 7 | no_helmet |
| 2 | vest    | | 8 | no_goggle |
| 3 | boots   | | 9 | no_gloves |
| 4 | goggles | | 10 | no_boots |
| 5 | none    | | | |

The six concepts we need (person, helmet, gloves, vest, boots, goggles) are all present.
The extra `no_*` classes (missing PPE) are useful later for compliance alerts.

## 10. Dataset validation

```cmd
python scripts\check_dataset.py
```

It checks folders, `data.yaml`, image/label counts per split, class IDs used, class
names, images without labels and labels without images, malformed label lines, and ends
with `RESULT: DATASET VALID` or a list of errors.

Known, harmless warning for this dataset: 10 extra label files in `labels/train`
(`image940(1).txt` etc.) have no matching image; YOLO simply ignores them.

## 11. Training the YOLO model

Settings are at the top of `training/train.py`:

```python
MODEL = "yolo26n.pt"   # YOLO26 nano - smallest/fastest current Ultralytics model
DATA = "dataset/data.yaml"
EPOCHS = 30
IMG_SIZE = 640
BATCH = 8              # reduce to 4 or 2 if you run out of memory
WORKERS = 2            # set to 0 if you get data-loader errors on Windows
DEVICE = None          # auto: GPU if available, otherwise CPU
```

Start training:
```cmd
python training\train.py
```
Override without editing the file, e.g. `python training\train.py --batch 4 --epochs 20`.

Quick 1-minute smoke test (checks everything works, the model will NOT be good):
```cmd
python training\train.py --epochs 1 --fraction 0.05 --imgsz 320 --name smoke_test
```

Equivalent Ultralytics CLI command:
```cmd
yolo detect train model=yolo26n.pt data=dataset/data.yaml epochs=30 imgsz=640 batch=8 project=runs/detect name=ppe_training
```

Expected time: on a laptop CPU roughly 3-8 minutes per epoch (30 epochs ≈ 2-4 hours);
with an NVIDIA GPU usually well under 1 minute per epoch. Training stops early if the
validation score stops improving for 10 epochs.

Output:
```
runs/
└── detect/
    └── ppe_training/
        ├── results.png, confusion_matrix.png, ...
        └── weights/
            ├── best.pt   <- use this as the production PPE model
            └── last.pt
```

If training is interrupted, resume with:
`yolo detect train resume model=runs/detect/ppe_training/weights/last.pt`

## 12. Testing the trained model

```cmd
python scripts\test_model.py --image path\to\test.jpg
python scripts\test_model.py --image path\to\test.jpg --conf 0.4 --show
```
Prints each detected class with its confidence and saves the annotated image to
`runs/predict/<name>_result.jpg`. If `best.pt` does not exist it prints
*"Trained model not found. Please train the model first."*

Tip: images in `dataset\images\test\` were not used for training, so they are good test
images.

## 13. Testing the webcam

```cmd
python scripts\test_camera.py
```
A window shows the camera feed; press **Q** to quit. Use `--camera 1` to try another
camera. (No PPE detection on the camera yet.)

## 14. Troubleshooting

| Problem | Fix |
|---------|-----|
| `python` not recognised | Reinstall Python with "Add to PATH" ticked, or use `py` instead of `python`. |
| `where python` shows a path outside `.venv` first | The venv is not active - run the activate command again in this terminal. |
| PowerShell: "running scripts is disabled" | `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| `ModuleNotFoundError: No module named 'ultralytics'` (or cv2/torch) | Activate `.venv`, then `pip install -r requirements.txt`. |
| `CUDA available: False` but you have an NVIDIA GPU | Install the CUDA PyTorch build (section 6). |
| Out of memory / training freezes | Lower `BATCH` (4 or 2) and/or `IMG_SIZE` (480) in `training/train.py`. |
| Windows data-loader / "BrokenPipe" errors | Set `WORKERS = 0` in `training/train.py`. |
| `Could not open camera 0` | Close Zoom/Teams, allow camera access in Windows Settings > Privacy > Camera, or try `--camera 1`. |
| `DATASET NOT FOUND` | Run `python scripts\download_dataset.py`. |
| `Trained model not found` | Train first: `python training\train.py`. |

## 15. Git notes

`.gitignore` excludes `.venv/`, `runs/`, `*.pt`, images and the downloaded dataset
contents, but keeps `dataset/data.yaml`. To submit the trained model, remove the `*.pt`
line from `.gitignore` (or copy `best.pt` into `models/` and add it with `git add -f`).
