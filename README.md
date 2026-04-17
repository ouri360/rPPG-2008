# 🫀 ICA-rPPG-2010: Real-Time Remote Photoplethysmography using the ICA algorithm.

> Blind Source Separation for contactless Heart Rate estimation — ICA untangles the pulse from noise.

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9.x-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-FastICA-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.21-FF6F00?style=flat-square)
![License](https://img.shields.io/badge/License-Research%20Only-lightgrey?style=flat-square)

---

## Overview

A professional-grade, real-time Python pipeline for contactless Heart Rate (HR) estimation using standard webcams or pre-recorded video datasets.

This repository implements a modernized version of the **Blind Source Separation (BSS)** rPPG method originally proposed by [Poh et al. (2010)](https://opg.optica.org/oe/fulltext.cfm?uri=oe-18-10-10762&id=199381). By applying **Independent Component Analysis (FastICA)** to the Red, Green, and Blue facial color channels, the pipeline mathematically untangles the cardiovascular pulse wave from independent noise sources such as ambient lighting flicker and subtle head motion.

---

## ⚙️ Pipeline & DSP Architecture

While rooted in the foundational 2010 ICA framework, this implementation applies several critical upgrades to address the known historical flaws of BSS algorithms.

### 1. Permutation Ambiguity Solver — Dynamic Spectral Selection
ICA is inherently "blind": it outputs components in a random, unpredictable order. Rather than hardcoding a component index, this pipeline runs a mini-FFT on all three unmixed components and dynamically selects the one with the maximum spectral power within the physiological band (0.7–3.0 Hz).

### 2. Hyper-Robust FastICA
The `scikit-learn` FastICA solver is explicitly parameterized with `max_iter=1000` to prevent convergence failure when subjects introduce sudden or heavy motion artifacts.

### 3. Area-Weighted Super Masks — `detector.py`
Uses MediaPipe Convex Hulls to map dense vascular regions (Forehead, Left Cheek, Right Cheek) onto a single unified mask. This automatically enforces a dynamic, physically area-weighted spatial average across all ROIs.

### 4. Hardware-Level V4L2 Locking — `webcam.py`
Bypasses the GStreamer middleware via the `cv2.CAP_V4L2` backend to interface directly with the Linux Kernel. Forcefully disables Auto-Exposure, Auto-White Balance, and Auto-Focus to eliminate camera firmware hunting as a noise source.

### 5. Decoupled Asynchronous Processing
The heavy FastICA matrix computation and the Matplotlib GUI are strictly decoupled, executing once every 15 frames. This unblocks the Python GIL, allowing the spatial extraction loop to maintain a stable 30 FPS.

### 6. Context-Aware, VFR-Immune Timelines
- **Live mode** — uses the atomic system clock (`time.time()`) to guard against dropped frames.
- **Dataset mode** — parses exact microsecond ground truth arrays (UBFC-rPPG) and reconstructs Variable Frame Rate MP4s onto a perfect 30 Hz grid via linear interpolation.

---

## 🗂️ Project Structure

| File | Responsibility |
|------|---------------|
| `main.py` | Decoupled orchestrator. Manages the async processing loop and the 3-panel Matplotlib dashboard. |
| `processor.py` | DSP engine. Handles time-grid interpolation, detrending, FastICA, 71-tap FIR filtering (`firwin`), and parabolic sub-bin FFT interpolation. |
| `detector.py` | Spatial extractor. Uses MediaPipe Face Mesh to track high-density vascular landmarks, pushing spatial noise to 30 Hz. |
| `webcam.py` | Hardware wrapper. Executes V4L2 parameter locks to guarantee a stable, dumb-sensor video feed. |
| `gt.py` | Dataset parser. Synchronizes UBFC-rPPG pulse oximeter ground truth files with the video feed. |

---

## 🚀 Installation

This pipeline requires a specific **"Golden Stack"** of dependencies to avoid C-API and backend routing conflicts.

> ⚠️ **NumPy must be < 2.0.0** for MediaPipe 0.10.x compatibility.

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/rppg-2010.git
cd rppg-2010
```

**2. Install exact dependencies**
```bash
pip install numpy==1.26.4 opencv-python==4.9.0.80 mediapipe==0.10.21 scipy matplotlib scikit-learn
```

---

## 💻 Usage

### Live Webcam

In `main.py`, set the video source to your camera index:
```python
VIDEO_SOURCE = 0
```
Then run:
```bash
python main.py
```

### 💾 UBFC-rPPG Dataset

Download a subject from the [UBFC-rPPG dataset](https://sites.google.com/view/ybenezeth/ubfcrppg) and point the pipeline to its files:
```python
VIDEO_SOURCE = "dataset/subject1.mp4"
GT_FILE      = "dataset/gt_subject1.txt"
```
Then run:
```bash
python main.py
```
The OpenCV feed will overlay the estimated ICA heartbeat against the medical ground truth in real time.

---

## 📊 Dashboard

When `DEBUG_MODE = True`, the application spawns a 3-panel asynchronous Matplotlib dashboard:

| Panel | Description |
|-------|-------------|
| **Raw RGB Signals** | Temporally uniform 1D projections of the R, G, and B channels prior to unmixing. |
| **Unmixed ICA Components** | All three blind sources separated by FastICA (semi-transparent), with the algorithmically verified physiological component highlighted in bold. |
| **Power Spectrum** | High-resolution FFT with parabolic sub-bin interpolation and BPM readout for the selected component. |

---

## 📄 References

- Poh, M. Z., McDuff, D. J., & Picard, R. W. (2010). Non-contact, automated cardiac pulse measurements using video imaging and blind source separation. *Optics Express*, 18(10), 10762–10774.
- Bobbia, S., Macwan, R., Benezeth, Y., Mansouri, A., & Dubois, J. (2019). Unsupervised skin tissue segmentation for remote photoplethysmography. *Pattern Recognition Letters*, 124, 82–90. *(UBFC-rPPG Dataset)*

---

> **Disclaimer:** This software is intended for research and educational purposes only. It is not designed or validated for medical diagnosis or clinical use.
