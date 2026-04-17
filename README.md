# 🫀 rPPG-2008: Real-Time Remote Photoplethysmography for NVIDIA Jetson Nano

> Lightweight, real-time Heart Rate estimation from a standard webcam — built for Edge AI.

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9.x-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.21-FF6F00?style=flat-square)
![Platform](https://img.shields.io/badge/Platform-PC%20%7C%20Jetson%20Nano-lightgrey?style=flat-square)
![License](https://img.shields.io/badge/License-Research%20Only-lightgrey?style=flat-square)

---

## Overview

An optimized, real-time implementation of the foundational remote photoplethysmography (rPPG) method proposed by [Verkruysse et al. (2008)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2717852/).

The pipeline extracts Heart Rate (BPM) from a standard webcam using computer vision and digital signal processing. It is architected to be exceptionally lightweight for **Edge AI environments** (e.g., Nvidia Jetson Nano), relying on the computationally inexpensive 1D Green channel while introducing robust mathematical defenses against real-world noise sources.

---

## ⚙️ Pipeline & DSP Architecture

While rooted in the 2008 paper, this implementation applies a series of hardened DSP upgrades to make the single-channel system reliable under real-world lighting and motion conditions.

### 1. Hardware Locking — `webcam.py`
Forcefully overrides the camera firmware to disable Auto-Exposure and Auto-White Balance, preventing the camera from injecting artificial periodic signals into the measurement.

### 2. Jitter-Free Tracking — `detector.py`
Uses **MediaPipe Face Mesh** to extract high-density vascular regions across three ROIs: Forehead, Left Cheek, and Right Cheek. **Exponential Moving Average (EMA) smoothing** is applied to raw facial landmarks to suppress high-frequency AI spatial micro-jitter.

### 3. Signal Extraction — `processor.py`

**Asymmetric Pixel Sorting** isolates the Green channel and applies a Trimmed Mean. The top 20% of pixels (specular glare, oil reflection) and bottom 5% (shadows, hair) are mathematically discarded, leaving only the diffuse reflection component.

**Dynamic ROI Weighting** fuses the three regions into a single 1D array using a 60/20/20 weighted average (60% Forehead, 20% Left Cheek, 20% Right Cheek), capitalizing on the higher vascular density of the forehead.

### 4. Pre-Processing & Shock Absorption

**Velocity Clamping** detects instantaneous vertical discontinuities (dropped frames, blinks) and clamps their derivatives, converting sharp signal cliffs into smooth slopes to prevent downstream filter ringing.

**Detrending & Normalization** applies linear detrending and variance standardization to remove baseline wander caused by slow postural drift.

### 5. Filtering & Frequency Analysis

A **Butterworth Bandpass filter** (0.7–3.0 Hz / 42–180 BPM) is designed via `scipy.signal.butter` and applied with `scipy.signal.sosfiltfilt` — forward then backward — for zero phase distortion.

**High-Resolution FFT** applies a Hanning window to the time-domain signal and computes the power spectrum. **Parabolic Sub-Bin Interpolation** is applied around the spectral peak to achieve BPM precision beyond the native FFT bin resolution.

---

## 🗂️ Project Structure

```
rPPG-2008/
├── main.py          # Entry point & Matplotlib visualization dashboard
├── processor.py     # DSP engine — sorting, velocity clamping, Butterworth, FFT
├── detector.py      # Vision processing — MediaPipe Face Mesh & EMA smoothing
├── webcam.py        # Hardware interface — OpenCV context manager & V4L2 params
├── gt.py            # Dataset parser — UBFC-rPPG ground truth synchronization
├── check.py         # Environment validator for dependency safety
└── README.md
```

---

## 🚀 Installation

This pipeline requires a specific **"Golden Stack"** of dependencies to avoid C-API and backend routing conflicts.

> ⚠️ **NumPy must be < 2.0.0** for MediaPipe 0.10.x compatibility.

### Option A — Standard PC / Laptop

```bash
git clone https://github.com/YOUR_USERNAME/rPPG-2008.git
cd rPPG-2008
pip install numpy==1.26.4 opencv-python==4.9.0.80 mediapipe==0.10.21 scipy matplotlib
```

Validate your environment before running:
```bash
python check.py
```

### Option B — Nvidia Jetson Nano (Edge Deployment)

> ⚠️ **Do not** use `pip install opencv-python` on the Jetson Nano. It will overwrite Nvidia's hardware-accelerated JetPack binaries. Use `apt` for heavy math libraries on ARM64.

```bash
git clone https://github.com/YOUR_USERNAME/rPPG-2008.git
cd rPPG-2008
sudo apt-get update
sudo apt-get install python3-scipy python3-numpy python3-matplotlib
pip install mediapipe==0.10.21
```

---

## 💻 Usage

```bash
python main.py
```

**Switching video source** — edit the top of `main.py`:
```python
VIDEO_SOURCE = 0                       # Live webcam
VIDEO_SOURCE = "dataset/subject1.mp4"  # Pre-recorded dataset
```

**`DEBUG_MODE` flag** controls the visualization layer:

| Mode | Behaviour | Recommended for |
|------|-----------|-----------------|
| `True` | Spawns a 3-panel live Matplotlib dashboard (Raw Signal, Filtered Signal, FFT Power Spectrum) | PC / Laptop |
| `False` | Disables Matplotlib entirely; BPM is rendered directly onto the OpenCV frame | Jetson Nano |

---

## 📊 Dashboard

When `DEBUG_MODE = True`, the 3-panel dashboard displays:

| Panel | Description |
|-------|-------------|
| **Raw Signal** | Normalized 1D Green channel projection after area-weighted ROI fusion. |
| **Filtered Signal** | Detrended, velocity-clamped, bandpass-filtered (0.7–3.0 Hz) pulse wave. |
| **Power Spectrum** | FFT power spectrum with parabolic peak interpolation and BPM readout. |

---

## 📄 References

- Verkruysse, W., Svaasand, L. O., & Nelson, J. S. (2008). Remote plethysmographic imaging using ambient light. *Optics Express*, 16(26), 21434–21445.
- Bobbia, S., Macwan, R., Benezeth, Y., Mansouri, A., & Dubois, J. (2019). Unsupervised skin tissue segmentation for remote photoplethysmography. *Pattern Recognition Letters*, 124, 82–90. *(UBFC-rPPG Dataset)*

---

> **Disclaimer:** This software is intended for research and educational purposes only. It is not designed or validated for medical diagnosis or clinical use.
