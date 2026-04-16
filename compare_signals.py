import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def compare_npz_signals(directory: str = "rppg_test/signals"):
    """
    Loads all .npz files from the specified directory and plots them 
    overlaid for direct comparison of time and frequency domains.
    """
    # Find all .npz files in the target directory
    filepaths = glob.glob(os.path.join(directory, "*.npz"))
    
    if not filepaths:
        print(f"No .npz files found in '{directory}'. Please run main.py first.")
        return
        
    print(f"Found {len(filepaths)} signal files. Generating comparison plot...")

    # Create a 2-panel figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.tight_layout(pad=5.0)
    
    for filepath in filepaths:
        # Extract a clean name for the legend
        filename = os.path.basename(filepath)
        # Simplify name if it has long timestamps (optional, adjust to taste)
        label_name = filename.replace('.npz', '')
        
        # Load the compressed arrays
        try:
            with np.load(filepath) as data:
                filtered_signal = data['filtered_signal']
                fft_freqs = data['fft_freqs']
                fft_power = data['fft_power']
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue
            
        # 1. Plot Time Domain (The Waveform)
        ax1.plot(filtered_signal, label=label_name, linewidth=1.5, alpha=0.8)
        
        # 2. Plot Frequency Domain (The FFT Peaks)
        ax2.plot(fft_freqs, fft_power, label=label_name, linewidth=1.5, alpha=0.8)
        
    # --- Format Time Domain Plot ---
    ax1.set_title("Time Domain: Filtered Signal Comparison")
    ax1.set_xlabel("Frames (Samples)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper right', fontsize='small')
    
    # --- Format Frequency Domain Plot ---
    ax2.set_title("Frequency Domain: Power Spectrum (FFT) Comparison")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Power")
    # Restrict X-axis to the human heart rate range (approx. 45 - 180 BPM)
    ax2.set_xlim([0.7, 3.0]) 
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='upper right', fontsize='small')
    
    plt.show()

if __name__ == "__main__":
    # Ensure this points to the exact folder where your logger saves the .npz files
    compare_npz_signals(directory="rppg_test/signals")