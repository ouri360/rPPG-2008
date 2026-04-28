"""
rPPG Benchmarking & Evaluation Tool
---------------------------------------
Runs the rPPG pipeline silently over a dataset, compares the estimated BPM 
to the medical ground truth every 15 frames, and calculates academic metrics.
"""

import cv2
import time
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import logging

from webcam import WebcamStream
from detector import FaceDetector
from processor import SignalProcessor
from gt import GroundTruthReader

# Suppress debug logs from the other modules to keep the terminal clean
logging.getLogger().setLevel(logging.WARNING)

def calculate_metrics(df: pd.DataFrame, output_csv: str):
    """Calculates academic metrics for rPPG evaluation and appends to CSV."""
    if len(df) == 0:
        print("No data collected. Video might be too short.")
        return

    # Extract the arrays
    est_bpm = df['Estimated_BPM'].values
    true_bpm = df['True_BPM'].values

    # 1. Mean Absolute Error (MAE)
    mae = np.mean(np.abs(est_bpm - true_bpm))

    # 2. Root Mean Square Error (RMSE)
    rmse = np.sqrt(np.mean((est_bpm - true_bpm)**2))

    # 3. Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((true_bpm - est_bpm) / true_bpm)) * 100

    # 4. Pearson Correlation Coefficient (r)
    # (Requires at least 2 points and variance to calculate)
    if len(df) > 1 and np.std(est_bpm) > 0 and np.std(true_bpm) > 0:
        pearson_r, _ = pearsonr(est_bpm, true_bpm)
    else:
        pearson_r = 0.0

    print("\n" + "="*40)
    print("BENCHMARK RESULTS")
    print("="*40)
    print(f"Total Valid Samples : {len(df)}")
    print(f"MAE (Error)         : {mae:.2f} BPM")
    print(f"RMSE (Variance)     : {rmse:.2f} BPM")
    print(f"MAPE (Percentage)   : {mape:.2f} %")
    print(f"Pearson (r)         : {pearson_r:.3f}")
    print("="*40)
    # --- Append to the bottom of the CSV ---
    with open(output_csv, 'a') as f:
        f.write("\n")
        f.write("=== BENCHMARK SUMMARY ===\n")
        f.write(f"Total Valid Samples,{len(df)}\n")
        f.write(f"MAE (BPM),{mae:.2f}\n")
        f.write(f"RMSE (BPM),{rmse:.2f}\n")
        f.write(f"MAPE (%),{mape:.2f}\n")
        f.write(f"Pearson (r),{pearson_r:.3f}\n")
        
    print(f"✅ Summary metrics successfully appended to {output_csv}")


def main():
    # --- CONFIGURATION ---
    VIDEO_SOURCE = "dataset/UBFC-rPPG-Set2-Realistic/vid_subject3.avi"       # Point this to your UBFC video
    GT_FILE = "dataset/UBFC-rPPG-Set2-Realistic/gt_subject3.txt"    # Point this to your UBFC ground truth
    OUTPUT_CSV = "dataset/results/benchmark_results_subject3_POS_DeepLearning_6.csv"
    WARMUP_SECONDS = 35.0                  # Ignore the first 35 seconds
    # ---------------------

    print(f"Starting benchmark on {VIDEO_SOURCE}...")
    print(f"Warming up buffer for {WARMUP_SECONDS} seconds...")

    detector = FaceDetector()
    gt_reader = GroundTruthReader(GT_FILE)
    
    results = []

    with WebcamStream(source=VIDEO_SOURCE) as cam:
        processor = SignalProcessor(buffer_seconds=30, target_fps=cam.fps)
        frame_counter = 0

        while True:
            success, frame = cam.read_frame()
            if not success:
                break
                
            frame_counter += 1
            
            # Use the CFR synthetic timeline for datasets
            timestamp = frame_counter / cam.fps 

            multi_rois = detector.get_face_mesh_rois(frame)

            if multi_rois:
                processor.extract_and_buffer_multi(frame, multi_rois, timestamp)

                # Only run the heavy math and save data every 15 frames
                if frame_counter % 15 == 0:
                    # Catch the output dynamically (works for both POS/ICA and Green Channel)
                    estimation_output = processor.estimate_heart_rate()
                    
                    if estimation_output is not None:
                        estimated_bpm = estimation_output[0]
                        
                        # Only start saving data AFTER the 40-second warm-up
                        if timestamp > WARMUP_SECONDS and estimated_bpm is not None:
                            true_bpm = gt_reader.get_hr_at_time(timestamp)
                            
                            if true_bpm is not None:
                                results.append({
                                    "Time (s)": round(timestamp, 2),
                                    "Estimated_BPM": round(estimated_bpm, 2),
                                    "True_BPM": round(true_bpm, 2)
                                })
                                
                                # Print progress to terminal
                                print(f"Time: {timestamp:.1f}s | Est: {estimated_bpm:.1f} | True: {true_bpm:.1f}")

    # Export the raw data so you can make your own graphs in Excel/Python later
    if len(results) > 0:
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nRaw data saved to {OUTPUT_CSV}")
        
        # Calculate and print the final SOTA metrics
        calculate_metrics(df, OUTPUT_CSV)
    else:
        print("\nBenchmark failed: Not enough data collected. Ensure the video is longer than 35 seconds.")

if __name__ == "__main__":
    main()