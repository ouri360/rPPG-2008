import numpy as np
import matplotlib.pyplot as plt
with open("dataset/gt_subject1.txt", "r") as f:
    lines = f.readlines()

ppg = np.array([float(x) for x in lines[0].split()])
hr = np.array([float(x) for x in lines[1].split()])
time = np.array([float(x) for x in lines[2].split()])


fft_ppg = np.fft.fftfreq(len(ppg), d=(time[1] - time[0]))
plt.plot(fft_ppg)
plt.show()