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
    True Hybrid Physics-Informed Neural Network for rPPG.
    The AI acts as a smart estimator for spatial weights and the POS Alpha value,
    while the final output rigidly respects the optical physics equation.
    """

    def __init__(self, num_rois: int = 9) -> None:
        super().__init__()
        
        # 1. Attention Spatiale: Reçoit l'écart-type (Standard Deviation) de S1 et S2
        self.spatial_attention = nn.Sequential(
            # THE FIX: This amplifier forces the AI to look at the dynamic noise!
            nn.BatchNorm1d(num_rois * 2), 
            nn.Linear(num_rois * 2, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_rois),
            nn.Softmax(dim=-1)
        )
        
        # 2. Estimateur d'Alpha
        self.alpha_estimator = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1), 
            nn.Flatten(),
            nn.Linear(16, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 1), 
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        B, C, L, R = x.shape
        
        # --- THE FIX: Use standard deviation instead of variance ---
        # std keeps the noise on the same scale as the signal, preventing vanishing features.
        std_features = torch.std(x, dim=2).view(B, -1) 
        attention_weights = self.spatial_attention(std_features) 
        
        # Save the weights for the UI Visualizer
        self.latest_weights = attention_weights.detach().cpu().numpy()
        
        weights = attention_weights.unsqueeze(1).unsqueeze(2)
        fused_signals = torch.sum(x * weights, dim=-1) 
        
        alpha = self.alpha_estimator(fused_signals) 

        alpha = alpha * 3.0 

        alpha = alpha.unsqueeze(2) 
        
        s1 = fused_signals[:, 0:1, :] 
        s2 = fused_signals[:, 1:2, :] 
        pulse = s1 + alpha * s2 
        
        return pulse.squeeze(1)