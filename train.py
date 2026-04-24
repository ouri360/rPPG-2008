"""
Training Script for POSNet
---------------------------------------
Handles the training loop, loss calculation, and model checkpointing.
"""

import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch import Tensor

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
        return 1.0 - torch.mean(pearson)


class MockRPPGDataset(Dataset):
    """
    Placeholder Dataset.
    This must be replaced with a class that actually reads your video frames 
    and ground truth CSVs.
    """

    def __init__(self, num_samples: int = 1000, seq_len: int = 48) -> None:
        self.seq_len = seq_len
        self.x_data = torch.randn(num_samples, 2, seq_len, 3, dtype=torch.float32)
        self.y_data = torch.sin(torch.linspace(0, 10, seq_len)).unsqueeze(0).repeat(num_samples, 1)

    def __len__(self) -> int:
        return len(self.x_data)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.x_data[idx], self.y_data[idx]


def train_model() -> None:
    """Main training loop."""
    # Hyperparameters
    epochs = 50
    batch_size = 32
    learning_rate = 1e-3
    seq_len = 48  # L = int(30fps * 1.6s)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Starting training on device: {device}")

    # Initialization
    model = POSNet(num_rois=3).to(device)
    criterion = NegativePearsonLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Note: Replace MockRPPGDataset with your actual UBFC dataloader
    dataset = MockRPPGDataset(num_samples=2000, seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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
    save_path = 'pos_net_weights.pth'
    torch.save(model.state_dict(), save_path)
    logging.info(f"Training complete. Weights saved to {save_path}.")


if __name__ == "__main__":
    train_model()