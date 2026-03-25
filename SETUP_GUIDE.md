# ASD Early Screening Tool — Setup Guide
### STAT Digitization Project

---

## 📋 Overview

This project digitizes the **Screening Tool for Autism in Toddlers (STAT)** using:
- **MobileNetV2** visual feature extraction
- **MediaPipe FaceMesh** eye/iris tracking
- **Librosa** speech prosody analysis
- **Tkinter** GUI

**Target accuracy: 85–90%** (requires fine-tuning on labeled ASD dataset)

---

## 🖥️ System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Windows 10 / macOS 12 / Ubuntu 20.04 | Windows 11 / Ubuntu 22.04 |
| Python | 3.9 | 3.10 or 3.11 |
| RAM | 8 GB | 16 GB |
| Camera | Any USB/built-in webcam | 1080p webcam |
| Microphone | Built-in or USB | External USB mic |
| GPU | Optional (CPU works) | NVIDIA CUDA GPU |

---

## ⚙️ Installation

### Step 1: Clone/Extract Project

```bash
# If using the zip file:
unzip asd_screening.zip
cd asd_screening
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note on PyAudio (Windows):** If `pip install pyaudio` fails, download the wheel manually:
> https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
> Then: `pip install PyAudio-0.2.14-cp311-cp311-win_amd64.whl`

> **Note on PyAudio (Linux):**
> ```bash
> sudo apt-get install portaudio19-dev python3-dev
> pip install pyaudio
> ```

### Step 4: Verify Installation

```bash
python -c "import cv2, mediapipe, torch, librosa; print('All modules OK')"
```

---

## 🚀 Running the Application

### Full GUI Application (Recommended)

```bash
python main.py
```

### Command-Line Demo (for testing without full GUI)

```bash
# Test gaze tracking only
python demo.py --mode gaze --duration 30

# Test speech analysis only
python demo.py --mode speech

# Full pipeline demo
python demo.py --mode full
```

### Gaze Calibration (Optional but Recommended)

Run before screening for better accuracy:

```bash
python utils/calibrate_gaze.py
```

---

## 🧠 Training the Model (For 85–90% Accuracy)

The model ships with pretrained ImageNet weights. To reach the target accuracy,
you need to **fine-tune on an ASD behavioral dataset**.

### Step 1: Acquire Dataset

**Option A — Kaggle ASD Dataset:**
```
https://www.kaggle.com/datasets/gpiosenka/autistic-children-data-set-traintestvalidate
```
Download and extract to `data/images/`

**Option B — Eye Tracking Dataset (from PMC11719697):**
The paper uses the ETSDS (Eye Tracking Scanpath Dataset). Request access from:
```
Yarmouk University, Computer Science Department
Contact: mwedyan@yu.edu.jo
```

**Option C — TalkBank Speech Dataset (for speech module):**
```
https://talkbank.org/ → CHILDES → ASD corpora
```
Used by the Nature paper (s41598-025-01500-6) for MLU/MLT features.

**Option D — Autism Brain Imaging Data Exchange (ABIDE):**
```
http://fcon_1000.projects.nitrc.org/indi/abide/
```

### Step 2: Organize Dataset

```
data/
  images/
    asd/
      child_001.jpg
      child_002.jpg
      ...
    non_asd/
      child_100.jpg
      child_101.jpg
      ...
```

### Step 3: Train

```bash
python train_model.py \
  --data_dir ./data/images \
  --epochs 30 \
  --batch_size 16 \
  --lr 0.001 \
  --save_dir ./models/checkpoints
```

**Expected training time:**
- CPU: ~2-4 hours (30 epochs)
- GPU (CUDA): ~20-40 minutes

### Step 4: Use Trained Model

After training, update `models/mobilenet_asd.py` `VisualProcessor.__init__`:
```python
self.load_model('./models/checkpoints/best_model_accX.XXX.pth')
```

---

## 📁 Project Structure

```
asd_screening/
├── main.py                    # Entry point (launches GUI)
├── demo.py                    # CLI demo for testing
├── train_model.py             # Model training script
├── requirements.txt
│
├── gui/
│   ├── __init__.py
│   └── app.py                 # Main Tkinter GUI (all screens)
│
├── modules/
│   ├── __init__.py
│   ├── gaze_tracker.py        # MediaPipe iris tracking + dot animator
│   └── speech_analyzer.py    # Prosody + rhythm feature extraction
│
├── models/
│   ├── __init__.py
│   └── mobilenet_asd.py       # MobileNetV2 + Fusion Classifier
│
├── utils/
│   └── calibrate_gaze.py     # 5-point gaze calibration
│
├── data/                      # (create this) datasets + calibration
│   └── images/
│       ├── asd/
│       └── non_asd/
│
├── reports/                   # Auto-created, stores JSON reports
└── models/checkpoints/        # Auto-created, stores trained models
```

---

## 📊 How the Scoring Works

### Gaze Risk Score (0–100)
Computed from MediaPipe iris tracking:

| Feature | ASD Indicator | Weight |
|---------|--------------|--------|
| Following Ratio | Low = risk | 35% |
| Gaze Deviation | High = risk | 20% |
| Saccade Smoothness | Low = risk | 20% |
| Mutual Gaze | Low = risk | 15% |
| Gaze Variability | High = risk | 10% |

### Speech Risk Score (0–100)
Based on Nature paper (s41598-025-01500-6) features:

| Feature | ASD Indicator | Weight |
|---------|--------------|--------|
| Pitch Std Dev | Very low/high = risk | ~20% |
| Pause Ratio | >60% = risk | ~15% |
| Repetition Score | >0.85 = risk | ~20% |
| Speech Rate | <0.5 or >8 syl/s = risk | ~15% |
| Rhythm Regularity | Low = risk | ~10% |
| Jitter | High = risk | ~10% |
| Dynamic Range | Very low = risk | ~10% |

### Fused Score
```
Final Score = Gaze(35%) + Speech(35%) + Visual(30%)
```

### Risk Categories
| Score | Category | Action |
|-------|----------|--------|
| 0–29 | LOW RISK | Routine monitoring |
| 30–54 | MODERATE RISK | Consider referral |
| 55–74 | ELEVATED RISK | Recommend referral |
| 75–100 | HIGH RISK | Urgent referral |

---

## 🔬 Research References

1. **Gaze Tracking:** Jaradat et al. (2024). "Using Machine Learning to Diagnose Autism
   Based on Eye Tracking Technology." *Diagnostics*, 15(1), 66.
   DOI: 10.3390/diagnostics15010066
   → Used MobileNet + ensemble achieving 96–98% on ETSDS dataset

2. **Speech Analysis:** Assaf et al. (2025). "Screening autism spectrum disorder in
   children using machine learning on speech transcripts." *Scientific Reports*, 15, 34134.
   DOI: 10.1038/s41598-025-01500-6
   → MLU, MLT ratio features achieving >86% accuracy on TalkBank data

---

## ⚠️ Clinical Disclaimer

This tool is for **SCREENING and RESEARCH purposes only**. It does not constitute
a clinical diagnosis of Autism Spectrum Disorder (ASD). All results must be
reviewed by qualified clinical professionals. Do not make clinical decisions
based solely on this software.

---

## 🐛 Troubleshooting

**Camera not opening:**
```bash
# Test camera independently
python -c "import cv2; cap=cv2.VideoCapture(0); print(cap.isOpened())"
```

**MediaPipe errors:**
```bash
pip install --upgrade mediapipe
```

**Audio recording fails:**
```bash
# List available audio devices
python -c "import sounddevice as sd; print(sd.query_devices())"
```

**CUDA/GPU not detected:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
# If False, CPU will be used automatically (slower but functional)
```

**tkinter not found (Linux):**
```bash
sudo apt-get install python3-tk
```

---

## 📝 License

For research and educational use. Not for commercial deployment.
Refer all clinical use to qualified medical professionals.
