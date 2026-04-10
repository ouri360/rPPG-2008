**Implementation of an rPPG Method using Python**

**The scipy.signal Toolkit**

We will use three specific functions from scipy.signal:

- **butter()**: Designs an infinite impulse response (IIR) Butterworth filter. You give it the cutoff frequencies and the order, and it returns the b (numerator) and a (denominator) coefficients of the filter's transfer function.

- **filtfilt()**: Applies the filter forward, then backward. This is crucial for rPPG. Standard real-time filters introduce a phase shift (delaying the signal). filtfilt cancels out this phase shift, keeping your data perfectly aligned in time.

- **welch()**: Instead of computing a raw FFT, which is highly susceptible to random noise spikes, Welch's method splits the signal into overlapping segments, computes the periodogram for each, and averages them. For an autonomous, unmonitored system that needs to be robust against real-world noise, Welch's method is the industry standard over a raw FFT.

**Import needed :**
- OpenCV
- Numpy
- Matplotlib

- Logging
- Time
- Collections
- Typing
