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

from dataset import UBFCPhysDataset
from model import POSNet

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class NegativePearsonLoss(nn.Module):
    """
    Computes the Negative Pearson Correlation Coefficient.
    Used to compare the phase and frequency of two waveforms, ignoring scale.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            preds (Tensor): Predicted signals, shape (Batch, SeqLen).
            targets (Tensor): Ground truth signals, shape (Batch, SeqLen).
        """
        mean_p = torch.mean(preds, dim=1, keepdim=True)
        mean_t = torch.mean(targets, dim=1, keepdim=True)
        
        preds_centered = preds - mean_p
        targets_centered = targets - mean_t
        
        cov = torch.sum(preds_centered * targets_centered, dim=1)
        std_p = torch.sqrt(torch.sum(preds_centered ** 2, dim=1) + 1e-8)
        std_t = torch.sqrt(torch.sum(targets_centered ** 2, dim=1) + 1e-8)
        
        pearson = cov / (std_p * std_t)
        
        # Minimize the loss -> maximize correlation (1.0 is perfect correlation)
        return 1.0 - torch.mean(torch.abs(pearson))


def train_model() -> None:
    """Main training loop."""
    # Hyperparameters
    epochs = 80
    batch_size = 128
    learning_rate = 1e-3      
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Starting training on device: {device}")

    # Initialization
    model = POSNet(num_rois=3).to(device)
    criterion = NegativePearsonLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # FIXED: Renamed to clearly identify this as the dataset cache, not model weights
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
            "dataset/UBFC-Phys-S1/vid_s1_T1.avi",
            "dataset/UBFC-Phys-S1/vid_s1_T2.avi",
            "dataset/UBFC-Phys-S1/vid_s1_T3.avi",
            "dataset/UBFC-Phys-S2/vid_s2_T1.avi",
            "dataset/UBFC-Phys-S2/vid_s2_T2.avi",
            "dataset/UBFC-Phys-S2/vid_s2_T3.avi",
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
            "dataset/UBFC-rPPG-Set2-Realistic/vid_subject9.avi"
        ]

        gt_files = [
            "dataset/UBFC-Phys-S1/bvp_s1_T1.csv",
            "dataset/UBFC-Phys-S1/bvp_s1_T2.csv",
            "dataset/UBFC-Phys-S1/bvp_s1_T3.csv",
            "dataset/UBFC-Phys-S2/bvp_s2_T1.csv",
            "dataset/UBFC-Phys-S2/bvp_s2_T2.csv",
            "dataset/UBFC-Phys-S2/bvp_s2_T3.csv",
            "dataset/UBFC-Phys-S1/bvp_s1_T1.csv",
            "dataset/UBFC-Phys-S1/bvp_s1_T2.csv",
            "dataset/UBFC-Phys-S1/bvp_s1_T3.csv",
            "dataset/UBFC-Phys-S2/bvp_s2_T1.csv",
            "dataset/UBFC-Phys-S2/bvp_s2_T2.csv",
            "dataset/UBFC-Phys-S2/bvp_s2_T3.csv",
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
            "dataset/UBFC-rPPG-Set2-Realistic/gt_subject9.txt"
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
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        
        if (epoch + 1) % 5 == 0:
            logging.info(f"Epoch [{epoch + 1}/{epochs}] - Loss: {avg_loss:.4f}")

    # Save the trained weights
    save_path = 'pos_net_weights.pt'
    torch.save(model.state_dict(), save_path)
    logging.info(f"Training complete. Weights saved to {save_path}.")


if __name__ == "__main__":
    train_model()