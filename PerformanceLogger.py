import pandas as pd
import numpy as np
import os
from datetime import datetime

class PerformanceLogger:
    def __init__(self, output_dir: str = "evaluation_results"):
        """
        Initializes the logger and creates the output directories.
        """
        self.output_dir = output_dir
        self.signals_dir = os.path.join(output_dir, "signals")
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.signals_dir, exist_ok=True)
        
        # This list will hold the summary data for our tabular output
        self.summary_data = []

    def log_video_result(self, video_name: str, estimated_bpm: float, real_bpm: float, 
                         filtered_signal: np.ndarray, fft_freqs: np.ndarray, fft_power: np.ndarray):
        """
        Saves the heavy arrays as a NumPy file and records the BPMs for the summary table.
        """
        # 1. Calculate the error
        error_bpm = abs(estimated_bpm - real_bpm)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 2. Save the heavy arrays as a compressed .npz file
        signal_filename = f"{video_name}_{timestamp}.npz"
        signal_filepath = os.path.join(self.signals_dir, signal_filename)
        
        np.savez_compressed(
            signal_filepath,
            filtered_signal=filtered_signal,
            fft_freqs=fft_freqs,
            fft_power=fft_power
        )
        
        # 3. Append the scalar metrics to our summary list
        self.summary_data.append({
            "Video_Name": video_name,
            "Date_Processed": timestamp,
            "Estimated_BPM": round(estimated_bpm, 2),
            "Real_BPM": round(real_bpm, 2),
            "Absolute_Error": round(error_bpm, 2),
            "Signal_File": signal_filename
        })
        
        print(f"Logged data for {video_name}. Error: {error_bpm:.2f} BPM")

    def export_summary_table(self, filename: str = "bpm_comparison_summary.csv"):
        """
        Exports the recorded metrics into a CSV table for easy comparison.
        """
        if not self.summary_data:
            print("No data to export.")
            return
            
        df = pd.DataFrame(self.summary_data)
        
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Summary table successfully saved to: {filepath}")