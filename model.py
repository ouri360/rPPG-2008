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
    """POSNet: A lightweight 1D CNN for rPPG signal fusion and alpha estimation. Designed to replace the mathematical alpha-tuning step in the POS algorithm with a learnable model that can adapt to different subjects and conditions."""
    # Added the return_telemetry flag
    def __init__(self, num_rois: int = 9, return_telemetry: bool = False) -> None:
        """Initializes the POSNet model architecture, including the spatial attention mechanism and alpha estimator. The return_telemetry flag allows the model to output intermediate values for AI-driven dashboard visualization."""
        super().__init__()
        self.return_telemetry = return_telemetry
        
        self.spatial_attention = nn.Sequential(
            nn.BatchNorm1d(num_rois * 2), 
            nn.Linear(num_rois * 2, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_rois),
            nn.Softmax(dim=-1)
        )
        
        self.alpha_estimator = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1), 
            nn.Flatten(),
            nn.Linear(16, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 1), 
            nn.Softplus() # Locked to Iteration 6
        )

    def forward(self, x: Tensor):
        """Forward pass through the POSNet model. Takes a tensor of shape (B, 2, SeqLen, ROIs) and outputs the fused pulse signal. If return_telemetry is True, also returns the attention weights and alpha values for dashboard visualization."""
        B, C, L, R = x.shape
        
        std_features = torch.std(x, dim=2).view(B, -1) 
        attention_weights = self.spatial_attention(std_features) 
        
        if not self.return_telemetry:
            # Native PyTorch telemetry caching
            self.latest_weights = attention_weights.detach().cpu().numpy()
        
        weights = attention_weights.unsqueeze(1).unsqueeze(2)
        fused_signals = torch.sum(x * weights, dim=-1) 
        
        alpha = self.alpha_estimator(fused_signals) 
        
        if not self.return_telemetry:
            # Native PyTorch telemetry caching
            self.latest_alpha = alpha.detach().cpu().numpy()
            
        alpha = alpha.unsqueeze(2) 
        
        s1 = fused_signals[:, 0:1, :] 
        s2 = fused_signals[:, 1:2, :] 
        pulse = s1 + alpha * s2 
        
        # ONNX EXPORT PATH: Send the variables directly out of the network
        if self.return_telemetry:
            return pulse.squeeze(1), attention_weights, alpha.squeeze(2)
            
        return pulse.squeeze(1)