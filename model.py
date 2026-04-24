"""
POSNet Module
---------------------------------------
A lightweight 1D Convolutional Neural Network designed to replace the 
mathematical alpha-tuning step of the POS rPPG algorithm.
Optimized for Edge AI deployment (e.g., Nvidia Jetson).
"""

import torch
import torch.nn as nn
from torch import Tensor


class POSNet(nn.Module):
    """
    Neural network for dynamic spatial and temporal fusion of rPPG signals.
    """

    def __init__(self, num_rois: int = 3) -> None:
        """
        Initializes the POSNet architecture.
        
        Args:
            num_rois (int): Number of Regions of Interest (e.g., 3 for forehead, cheeks).
        """
        super().__init__()
        
        # Spatial Attention: Learns to dynamically weigh ROIs based on signal quality
        self.spatial_attention = nn.Sequential(
            nn.Linear(num_rois, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, num_rois),
            nn.Softmax(dim=-1)
        )
        
        # Temporal Convolution: Learns the non-linear combination of S1 and S2
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=16, out_channels=1, kernel_size=3, padding=1)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the network.
        
        Args:
            x (Tensor): Input tensor of shape (Batch, Channels=2, SeqLen, ROIs).
            
        Returns:
            Tensor: Predicted 1D PPG signal of shape (Batch, SeqLen).
        """
        # Calculate temporal variance to identify motion artifacts (Batch, Channels, ROIs)
        variance = torch.var(x, dim=2)
        mean_variance = torch.mean(variance, dim=1)  # Shape: (Batch, ROIs)
        
        # Generate attention weights (Batch, ROIs)
        attention_weights = self.spatial_attention(mean_variance)
        
        # Broadcast weights to match input dimensions: (Batch, 1, 1, ROIs)
        weights = attention_weights.unsqueeze(1).unsqueeze(2)
        
        # Apply spatial attention and sum across ROIs -> (Batch, Channels=2, SeqLen)
        x_spatial_fused = torch.sum(x * weights, dim=-1)
        
        # Apply temporal convolution to fuse S1 and S2 -> (Batch, 1, SeqLen)
        h = self.temporal_conv(x_spatial_fused)
        
        # Remove the redundant channel dimension -> (Batch, SeqLen)
        return h.squeeze(1)