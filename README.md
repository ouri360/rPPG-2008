# sota-pos-rppg

> Real-time, contactless Heart Rate estimation using pure POS matrix mathematics — no black boxes, no neural nets.

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9.x-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.21-FF6F00?style=flat-square)
![License](https://img.shields.io/badge/License-Research%20Only-lightgrey?style=flat-square)

---

## Overview

This repository is a professional-grade, real-time Python pipeline for contactless Heart Rate (HR) estimation using standard webcams or pre-recorded video datasets.

It implements a highly optimized, strictly mathematical version of the **POS (Plane-Orthogonal-to-Skin)** algorithm ([Wang et al., 2016](#references)). By treating the human face as a multi-channel RGB sensor, the pipeline computationally isolates the diffuse reflection of the cardiovascular pulse wave while mathematically annihilating specular glare, ambient lighting flicker, and motion artifacts.

---

## Features

### Strict POS Matrix Mathematics
Implements pure POS projection with Overlap-Add (OLA) stitching and *Inner Flat Slicing* to completely eliminate edge-amplification noise prior to detrending.

### Dynamic Area-Weighted Super Masks
Uses MediaPipe Convex Hulls to map dense vascular regions (forehead, cheeks). RGB channels are extracted via a unified area-weighted mean, preventing "Boiling Mask" quantization jitter on compressed MP4s.

### Hardware-Level V4L2 Locking
Bypasses the GStreamer middleware to interface directly with the Linux Kernel, forcefully disabling Auto-Exposure, Auto-White Balance, and Auto-Focus — eliminating the 2.5–3.0 Hz synthetic noise trap caused by camera firmware hunting.

### Decoupled Asynchronous Processing
The Matplotlib GUI and heavy FFT computations are strictly decoupled, running every 15 frames. This unblocks the Python GIL, allowing the camera loop and spatial extraction to maintain a stable 30 FPS.

### Context-Aware, VFR-Immune Timelines
- **Live mode** — uses the atomic system clock (`time.time()`) to guard against dropped frames.
- **Dataset mode** — parses exact microsecond ground truth arrays (UBFC-rPPG) and reconstructs Variable Frame Rate MP4s onto a perfect 30 Hz grid via linear interpolation.

---

## Architecture

| File | Responsibility |
|------|---------------|
| `main.py` | Decoupled orchestrator. Manages the async processing loop and the 3-panel Matplotlib dashboard. |
| `processor.py` | DSP engine. Handles time-grid interpolation, POS projections, OLA stitching, and Butterworth bandpass filtering (0.7–3.0 Hz). |
| `detector.py` | Spatial extractor. Uses MediaPipe Face Mesh to dynamically track high-density vascular landmarks without arbitrary temporal smoothing. |
| `webcam.py` | Hardware wrapper. Executes V4L2 parameter locks to guarantee a stable, dumb-sensor video feed. |
| `gt.py` | Dataset parser. Synchronizes UBFC-rPPG pulse oximeter ground truth files with the video feed. |
| `check.py` | Environment validator. Verifies the "Golden Stack" dependencies to prevent C-API and dependency crashes. |

---

## Installation

This pipeline relies on precise memory management and underlying C-APIs. A specific **"Golden Stack"** of dependencies is required.

> ⚠️ **Do not use NumPy 2.x+** — it breaks the MediaPipe 0.10.x backend.

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/pos-rppg-2026.git
cd pos-rppg-2026
```

**2. Install exact dependencies**
```bash
pip install numpy==1.26.4 opencv-python==4.9.0.80 mediapipe==0.10.21 scipy matplotlib scikit-learn
```

**3. Validate your environment**
```bash
python check.py
```
Expect a fully green output before proceeding.

---

## Usage

### Live Webcam

In `main.py`, set the video source to your camera index:
```python
VIDEO_SOURCE = 0
```
Then run:
```bash
python main.py
```

### UBFC-rPPG Dataset

Download a subject from the [UBFC-rPPG dataset](https://sites.google.com/view/ybenezeth/ubfcrppg) and point the pipeline to its files:
```python
VIDEO_SOURCE = "dataset/vid_subject1.mp4"
GT_FILE      = "dataset/gt_subject1.txt"
```
Then run:
```bash
python main.py
```
The dashboard will plot predicted HR against the medical ground truth in real time.

---

## Dashboard

When `DEBUG_MODE = True`, the application spawns a 3-panel asynchronous Matplotlib dashboard:

| Panel | Description |
|-------|-------------|
| **Raw Signal** | Normalized, temporally uniform 1D projection of the skin-tone channels. |
| **Filtered Signal** | Fully detrended and bandpass-filtered (0.7–3.0 Hz) time-domain pulse wave. |
| **Power Spectrum** | High-resolution FFT with peak detection, isolating the dominant physiological frequency. |

---

## References

- Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2016). Algorithmic principles of remote PPG. *IEEE Transactions on Biomedical Engineering*, 64(7), 1479–1491.
- Bobbia, S., Macwan, R., Benezeth, Y., Mansouri, A., & Dubois, J. (2019). Unsupervised skin tissue segmentation for remote photoplethysmography. *Pattern Recognition Letters*, 124, 82–90. *(UBFC-rPPG Dataset)*

---

> **Disclaimer:** This software is intended for research and educational purposes only. It is not designed or validated for medical diagnosis or clinical use.
