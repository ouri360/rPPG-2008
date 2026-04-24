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

    def __init__(self, num_rois: int = 3) -> None:
        super().__init__()
        
        # 1. Attention Spatiale: Reçoit la variance de S1 et S2 pour chaque ROI
        self.spatial_attention = nn.Sequential(
            nn.Linear(num_rois * 2, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, num_rois),
            nn.Softmax(dim=-1)
        )
        
        # 2. Estimateur d'Alpha: Un CNN qui analyse globalement la fenêtre
        self.alpha_estimator = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            # L'AdaptiveAvgPool1d est magique: il force le réseau à analyser 
            # l'intégralité de la fenêtre de 56 frames d'un seul coup (Global Context)
            nn.AdaptiveAvgPool1d(1), 
            nn.Flatten(),
            nn.Linear(16, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 1), # Prédit un SEUL scalaire : la valeur Alpha parfaite !
            nn.Softplus()
        )

    def forward(self, x: Tensor) -> Tensor:
        # x shape: (Batch, Channels=2, SeqLen, ROIs=3)
        B, C, L, R = x.shape
        
        # --- ETAPE 1 : FUSION SPATIALE INTELLIGENTE ---
        # On donne au réseau les variances de S1 et S2 pour chaque région
        var_features = torch.var(x, dim=2).view(B, -1) # Shape: (Batch, 6)
        attention_weights = self.spatial_attention(var_features) # Shape: (Batch, 3)
        
        # Appliquer les poids et fusionner les régions -> (Batch, Channels=2, SeqLen)
        weights = attention_weights.unsqueeze(1).unsqueeze(2)
        fused_signals = torch.sum(x * weights, dim=-1) 
        
        # --- ETAPE 2 : ESTIMATION NEURONALE DE L'ALPHA ---
        # Le réseau analyse le signal fusionné et prédit l'alpha optimal
        alpha = self.alpha_estimator(fused_signals) # Shape: (Batch, 1)
        alpha = alpha.unsqueeze(2) # Shape: (Batch, 1, 1) pour permettre la multiplication
        
        # --- ETAPE 3 : LA PHYSIQUE OPTIQUE (POS MATHS) ---
        s1 = fused_signals[:, 0:1, :] # Shape: (Batch, 1, SeqLen)
        s2 = fused_signals[:, 1:2, :] # Shape: (Batch, 1, SeqLen)
        
        # C'est la véritable équation POS, mais l'IA a trouvé l'alpha parfait
        # en analysant la morphologie du bruit au lieu d'une simple division de variance.
        pulse = s1 + alpha * s2 
        
        return pulse.squeeze(1) # Shape: (Batch, SeqLen)