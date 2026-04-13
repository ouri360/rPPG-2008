# 🫀 rPPG-2008: Real-Time Remote Photoplethysmography for NVIDIA Jetson Nano

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![SciPy](https://img.shields.io/badge/SciPy-DSP-red)
![Platform](https://img.shields.io/badge/Platform-PC%20%7C%20Jetson%20Nano-lightgrey)

An optimized, real-time implementation of the foundational remote photoplethysmography (rPPG) method proposed by [Verkruysse et al. (2008)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2717852/). 

This project extracts a human heart rate (BPM) from a standard webcam feed using computer vision and digital signal processing. It is specifically architected for **Edge AI environments** (e.g., Nvidia Jetson Nano), featuring robust noise reduction, memory-efficient rolling buffers, and a decoupled object-oriented design.

---

## ⚙️ Core Pipeline & DSP Architecture

While based on the 2008 paper, this implementation introduces several modern DSP upgrades to make the system robust under real-world lighting and motion conditions:

1. **Hardware Locking:** Disables webcam Auto-Exposure and Auto-White Balance to prevent artificial signal spikes.
2. **Facial Tracking (Viola-Jones):** Uses OpenCV Haar Cascades with **Exponential Moving Average (EMA) smoothing** on the bounding box to eliminate high-frequency spatial jitter.
3. **Signal Extraction:** Isolates the Region of Interest (ROI) and calculates the spatial average of the **Green Channel** (where hemoglobin absorption is highest) into a rolling time-series buffer.
4. **Pre-Processing (Detrending):** Applies linear detrending to eliminate baseline wander caused by slow postural movements or ambient light shifts, preventing IIR filter ringing.
5. **The `scipy.signal` Toolkit (Filtering & Frequency Analysis):**
   - **`butter()`**: Designs an infinite impulse response (IIR) Butterworth filter (Bandpass 0.7 Hz - 3.0 Hz). Returns the numerator and denominator coefficients of the filter's transfer function.
   - **`filtfilt()`**: Applies the filter forward, then backward. Standard real-time filters introduce a phase shift (delaying the signal). `filtfilt` cancels out this phase shift, keeping data perfectly aligned in time.
   - **`welch()`**: Instead of computing a raw FFT (which is highly susceptible to random noise spikes), Welch's method splits the signal into overlapping segments, computes the periodogram for each, and averages them. For an autonomous, unmonitored system that needs to be robust against real-world noise, Welch's method is the industry standard.

---

## 🗂️ Project Structure

The architecture is strictly modular to separate hardware interfacing, vision processing, and mathematical analysis.

```text
rPPG-2008/
│
├── camera.py        # Hardware interface (WebcamStream context manager)
├── detector.py      # Vision processing (FaceDetector & EMA ROI smoothing)
├── processor.py     # DSP math (SignalProcessor, Butterworth, Welch's PSD)
├── main.py          # Entry point & Matplotlib visualization dashboard
└── README.md
```

---

## 🚀 Installation & Setup
**Dependencies**

The project relies on the following standard and third-party libraries:

    Core: cv2 (OpenCV), numpy, scipy, matplotlib.pyplot

    Standard Python: logging, time, collections (deque), typing (Tuple, Optional, List)

**Option A: Standard PC / Laptop (Prototyping)**

For Windows/macOS/Linux x86 machines:
```text
git clone [https://github.com/ouri360/rPPG-2008.git](https://github.com/ouri360/rPPG-2008.git)
cd rPPG-2008
pip install opencv-python numpy scipy matplotlib
```
**Option B: Nvidia Jetson Nano (Edge Deployment)**

⚠️ **IMPORTANT**: Do not use pip install opencv-python on the Jetson Nano, as it will overwrite Nvidia's hardware-accelerated JetPack binaries. Use the apt package manager for heavy math libraries on ARM64 architectures.
```text
git clone [https://github.com/ouri360/rPPG-2008.git](https://github.com/ouri360/rPPG-2008.git)
cd rPPG-2008
sudo apt-get update
sudo apt-get install python3-scipy python3-numpy python3-matplotlib
```
---

## 💻 Usage

Run the main pipeline:
```text
python main.py
```
**The `DEBUG_MODE` Flag**

At the top of main.py, you will find a DEBUG_MODE boolean used to manage CPU resources depending on your hardware:

    DEBUG_MODE = True: (Recommended for PC) Launches a 3-panel live Matplotlib dashboard alongside the webcam feed, visualizing the Raw Signal, the Filtered Signal, and the Welch Frequency Spectrum.

    DEBUG_MODE = False: (Recommended for Jetson Nano) Disables the heavy Matplotlib GUI to preserve CPU cycles for DSP calculations. The BPM is rendered directly onto the lightweight OpenCV frame.    