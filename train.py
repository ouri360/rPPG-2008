"""
Training Script for POSNet
---------------------------------------
Handles the training loop, loss calculation, and model checkpointing.
"""

import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import Tensor
import torch.fft

from dataset import UBFCPhysDataset
from model import POSNet

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class HybridrPPGLoss(nn.Module):
    """
    State-of-the-Art rPPG Loss Function.
    Combines Phase-Invariant time-domain correlation with 
    Frequency-Domain Power Spectrum matching (SNR optimization).
    """
    def __init__(self, max_shift: int = 15, freq_weight: float = 0.5) -> None:
        super().__init__()
        self.max_shift = max_shift
        self.freq_weight = freq_weight

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # ==========================================
        # 1. TIME DOMAIN: Phase-Invariant Pearson
        # ==========================================
        best_pearson = -1.0 * torch.ones(preds.size(0), device=preds.device)
        
        # Center the signals
        mean_t = torch.mean(targets, dim=1, keepdim=True)
        targets_centered = targets - mean_t
        std_t = torch.sqrt(torch.sum(targets_centered ** 2, dim=1) + 1e-8)

        for shift in range(-self.max_shift, self.max_shift + 1):
            preds_shifted = torch.roll(preds, shifts=shift, dims=1)
            mean_p = torch.mean(preds_shifted, dim=1, keepdim=True)
            preds_centered = preds_shifted - mean_p
            std_p = torch.sqrt(torch.sum(preds_centered ** 2, dim=1) + 1e-8)
            
            cov = torch.sum(preds_centered * targets_centered, dim=1)
            pearson = cov / (std_p * std_t)
            best_pearson = torch.maximum(best_pearson, pearson)
            
        time_loss = 1.0 - torch.mean(best_pearson)

        # ==========================================
        # 2. FREQUENCY DOMAIN: Power Spectrum Matching
        # ==========================================
        # FIX 1: ALWAYS use the centered predictions for FFT to kill the DC baseline
        preds_for_fft = preds - torch.mean(preds, dim=1, keepdim=True)
        targets_for_fft = targets - torch.mean(targets, dim=1, keepdim=True)

        fft_preds = torch.fft.rfft(preds_for_fft, dim=1)
        fft_targets = torch.fft.rfft(targets_for_fft, dim=1)

        power_preds = torch.abs(fft_preds) ** 2
        power_targets = torch.abs(fft_targets) ** 2

        # FIX 2: Drop the 0 Hz (DC) bin completely so it doesn't skew the SNR!
        power_preds = power_preds[:, 1:]
        power_targets = power_targets[:, 1:]

        # Normalize into probability distributions
        prob_preds = power_preds / (torch.sum(power_preds, dim=1, keepdim=True) + 1e-8)
        prob_targets = power_targets / (torch.sum(power_targets, dim=1, keepdim=True) + 1e-8)

        # KL-Divergence Loss (Minimum score is exactly 0.0)
        # We subtract the target's own chaos so we are only grading the AI's mistakes.
        freq_loss = torch.sum(prob_targets * (torch.log(prob_targets + 1e-8) - torch.log(prob_preds + 1e-8)), dim=1)
        freq_loss = torch.mean(freq_loss)

        # ==========================================
        # 3. HYBRID FUSION
        # ==========================================
        total_loss = (1.0 - self.freq_weight) * time_loss + (self.freq_weight) * freq_loss
        
        return total_loss


def train_model() -> None:
    """Main training loop."""
    # Hyperparameters
    epochs = 80
    batch_size = 128
    learning_rate = 1e-3      
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Starting training on device: {device}")

    # Initialization
    model = POSNet(num_rois=9).to(device)
    criterion = HybridrPPGLoss(max_shift=15, freq_weight=0.2)

    # 1. Add weight_decay to prevent overfitting to noisy videos
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # 2. Add a scheduler to gently lower the learning rate over 80 epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # Identify this as the dataset cache, not model weights
    cache_path = 'preprocessed_ubfc_dataset.pt'

    if os.path.exists(cache_path):
        logging.info(f"Found cached dataset at '{cache_path}'. Loading directly into memory...")
        dataset = torch.load(cache_path, weights_only=False)
        logging.info("Dataset loaded instantly!")
        
    else:
        logging.info("No cache found. Processing videos from scratch (this may take a while)...")

        video_files = [
            "dataset/UBFC-Phys-S1/vid_s1_T1.avi",
            "dataset/UBFC-Phys-S1/vid_s1_T2.avi",
            "dataset/UBFC-Phys-S1/vid_s1_T3.avi",
            "dataset/UBFC-Phys-S2/vid_s2_T1.avi",
            "dataset/UBFC-Phys-S2/vid_s2_T2.avi",
            "dataset/UBFC-Phys-S2/vid_s2_T3.avi",
            "dataset/UBFC-Phys-S3/vid_s3_T1.avi",
            "dataset/UBFC-Phys-S3/vid_s3_T2.avi",
            "dataset/UBFC-Phys-S3/vid_s3_T3.avi",
            "dataset/UBFC-Phys-S4/vid_s4_T1.avi",
            "dataset/UBFC-Phys-S4/vid_s4_T2.avi",
            "dataset/UBFC-Phys-S4/vid_s4_T3.avi",
            "dataset/UBFC-Phys-S5/vid_s5_T1.avi",
            "dataset/UBFC-Phys-S5/vid_s5_T2.avi",
            "dataset/UBFC-Phys-S5/vid_s5_T3.avi",
            "dataset/UBFC-Phys-S6/vid_s6_T1.avi",
            "dataset/UBFC-Phys-S6/vid_s6_T2.avi",
            "dataset/UBFC-Phys-S6/vid_s6_T3.avi",
            "dataset/UBFC-Phys-S7/vid_s7_T1.avi",
            "dataset/UBFC-Phys-S7/vid_s7_T2.avi",
            "dataset/UBFC-Phys-S7/vid_s7_T3.avi",
            "dataset/UBFC-Phys-S8/vid_s8_T1.avi",
            "dataset/UBFC-Phys-S8/vid_s8_T2.avi",
            "dataset/UBFC-Phys-S8/vid_s8_T3.avi",
            "dataset/UBFC-Phys-S9/vid_s9_T1.avi",
            "dataset/UBFC-Phys-S9/vid_s9_T2.avi",
            "dataset/UBFC-Phys-S9/vid_s9_T3.avi",
            "dataset/UBFC-Phys-S10/vid_s10_T1.avi",
            "dataset/UBFC-Phys-S10/vid_s10_T2.avi",
            "dataset/UBFC-Phys-S10/vid_s10_T3.avi",
            "dataset/UBFC-Phys-S11/vid_s11_T1.avi",
            "dataset/UBFC-Phys-S11/vid_s11_T2.avi",
            "dataset/UBFC-Phys-S11/vid_s11_T3.avi",
            "dataset/UBFC-Phys-S12/vid_s12_T1.avi",
            "dataset/UBFC-Phys-S12/vid_s12_T2.avi",
            "dataset/UBFC-Phys-S12/vid_s12_T3.avi",
            "dataset/UBFC-rPPG-Set1-Not_Moving/vid_subject5.avi",
            "dataset/UBFC-rPPG-Set1-Not_Moving/vid_subject6.avi",
            "dataset/UBFC-rPPG-Set1-Not_Moving/vid_subject7.avi",
            "dataset/UBFC-rPPG-Set1-Not_Moving/vid_subject8.avi",
            "dataset/UBFC-rPPG-Set1-Not_Moving/vid_subject10.avi",
            "dataset/UBFC-rPPG-Set1-Not_Moving/vid_subject11.avi",
            "dataset/UBFC-rPPG-Set1-Not_Moving/vid_subject12.avi",
            "dataset/UBFC-rPPG-Set2-Realistic/vid_subject1.avi",
            "dataset/UBFC-rPPG-Set2-Realistic/vid_subject3.avi",
            "dataset/UBFC-rPPG-Set2-Realistic/vid_subject4.avi",
            "dataset/UBFC-rPPG-Set2-Realistic/vid_subject9.avi",
            "dataset/UBFC-rPPG-Set2-Realistic/vid_subject20.avi",
            "dataset/UBFC-rPPG-Set2-Realistic/vid_subject23.avi",
            "dataset/UBFC-rPPG-Set2-Realistic/vid_subject24.avi",
            "dataset/UBFC-rPPG-Set2-Realistic/vid_subject30.avi",
            "dataset/UBFC-rPPG-Set2-Realistic/vid_subject31.avi",
            "dataset/UBFC-rPPG-Set2-Realistic/vid_subject32.avi"
        ]

        gt_files = [
            "dataset/UBFC-Phys-S1/bvp_s1_T1.csv",
            "dataset/UBFC-Phys-S1/bvp_s1_T2.csv",
            "dataset/UBFC-Phys-S1/bvp_s1_T3.csv",
            "dataset/UBFC-Phys-S2/bvp_s2_T1.csv",
            "dataset/UBFC-Phys-S2/bvp_s2_T2.csv",
            "dataset/UBFC-Phys-S2/bvp_s2_T3.csv",
            "dataset/UBFC-Phys-S3/bvp_s3_T1.csv",
            "dataset/UBFC-Phys-S3/bvp_s3_T2.csv",
            "dataset/UBFC-Phys-S3/bvp_s3_T3.csv",
            "dataset/UBFC-Phys-S4/bvp_s4_T1.csv",
            "dataset/UBFC-Phys-S4/bvp_s4_T2.csv",
            "dataset/UBFC-Phys-S4/bvp_s4_T3.csv",
            "dataset/UBFC-Phys-S5/bvp_s5_T1.csv",
            "dataset/UBFC-Phys-S5/bvp_s5_T2.csv",
            "dataset/UBFC-Phys-S5/bvp_s5_T3.csv",
            "dataset/UBFC-Phys-S6/bvp_s6_T1.csv",
            "dataset/UBFC-Phys-S6/bvp_s6_T2.csv",
            "dataset/UBFC-Phys-S6/bvp_s6_T3.csv",
            "dataset/UBFC-Phys-S7/bvp_s7_T1.csv",
            "dataset/UBFC-Phys-S7/bvp_s7_T2.csv",
            "dataset/UBFC-Phys-S7/bvp_s7_T3.csv",
            "dataset/UBFC-Phys-S8/bvp_s8_T1.csv",
            "dataset/UBFC-Phys-S8/bvp_s8_T2.csv",
            "dataset/UBFC-Phys-S8/bvp_s8_T3.csv",
            "dataset/UBFC-Phys-S9/bvp_s9_T1.csv",
            "dataset/UBFC-Phys-S9/bvp_s9_T2.csv",
            "dataset/UBFC-Phys-S9/bvp_s9_T3.csv",
            "dataset/UBFC-Phys-S10/bvp_s10_T1.csv",
            "dataset/UBFC-Phys-S10/bvp_s10_T2.csv",
            "dataset/UBFC-Phys-S10/bvp_s10_T3.csv",
            "dataset/UBFC-Phys-S11/bvp_s11_T1.csv",
            "dataset/UBFC-Phys-S11/bvp_s11_T2.csv",
            "dataset/UBFC-Phys-S11/bvp_s11_T3.csv",
            "dataset/UBFC-Phys-S12/bvp_s12_T1.csv",
            "dataset/UBFC-Phys-S12/bvp_s12_T2.csv",
            "dataset/UBFC-Phys-S12/bvp_s12_T3.csv",
            "dataset/UBFC-rPPG-Set1-Not_Moving/gt_subject5.xmp",
            "dataset/UBFC-rPPG-Set1-Not_Moving/gt_subject6.xmp",
            "dataset/UBFC-rPPG-Set1-Not_Moving/gt_subject7.xmp",
            "dataset/UBFC-rPPG-Set1-Not_Moving/gt_subject8.xmp",
            "dataset/UBFC-rPPG-Set1-Not_Moving/gt_subject10.xmp",
            "dataset/UBFC-rPPG-Set1-Not_Moving/gt_subject11.xmp",
            "dataset/UBFC-rPPG-Set1-Not_Moving/gt_subject12.xmp",
            "dataset/UBFC-rPPG-Set2-Realistic/gt_subject1.txt",
            "dataset/UBFC-rPPG-Set2-Realistic/gt_subject3.txt",
            "dataset/UBFC-rPPG-Set2-Realistic/gt_subject4.txt",
            "dataset/UBFC-rPPG-Set2-Realistic/gt_subject9.txt",
            "dataset/UBFC-rPPG-Set2-Realistic/gt_subject20.txt",
            "dataset/UBFC-rPPG-Set2-Realistic/gt_subject23.txt",
            "dataset/UBFC-rPPG-Set2-Realistic/gt_subject24.txt",
            "dataset/UBFC-rPPG-Set2-Realistic/gt_subject30.txt",
            "dataset/UBFC-rPPG-Set2-Realistic/gt_subject31.txt",
            "dataset/UBFC-rPPG-Set2-Realistic/gt_subject32.txt"
        ]

        # Initialize the actual dataset
        dataset = UBFCPhysDataset(
            video_paths=video_files, 
            gt_paths=gt_files
        )

        # Save the processed dataset to disk for next time
        logging.info(f"Saving processed dataset to '{cache_path}' for future runs...")
        torch.save(dataset, cache_path)
        logging.info("Cache saved successfully!")

    # Retrieve the auto-calculated parameters (works whether cached or freshly built)
    dynamic_seq_len = dataset.seq_len
    native_fps = dataset.target_fps
    logging.info(f"Training configured for {native_fps} FPS with sequence length {dynamic_seq_len}") 

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    # Training Loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            preds = model(batch_x)
            
            # Calculate loss
            loss = criterion(preds, batch_y)
            
            # Backward pass and optimize
            loss.backward()

            # UPGRADE: Gradient Clipping to protect against sudden motion spikes
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        scheduler.step()
        
        if (epoch + 1) % 5 == 0:
            logging.info(f"Epoch [{epoch + 1}/{epochs}] - Loss: {avg_loss:.4f}")

    # Save the trained weights
    save_path = 'pos_net_weights.pt'
    torch.save(model.state_dict(), save_path)
    logging.info(f"Training complete. Weights saved to {save_path}.")


if __name__ == "__main__":
    train_model()