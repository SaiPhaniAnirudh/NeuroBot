"""
EEGNet-8,2 (Lawhern et al. 2018) — the base paper's primary decoder.
F1=8 temporal filters, D=2 depth multiplier, F2=16 pointwise filters.

Input: (batch, n_channels, n_samples) or (batch, 1, n_channels, n_samples)
"""

import torch
import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(self, n_channels: int, n_samples: int, n_classes: int,
                 F1: int = 8, D: int = 2, F2: int = 16, kernel_length: int = 64,
                 dropout: float = 0.5):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),  # depthwise spatial
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),  # depthwise
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),  # pointwise
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            flat_dim = self.block2(self.block1(dummy)).numel()

        self.classifier = nn.Linear(flat_dim, n_classes)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = x.flatten(1)
        return self.classifier(x)

    def frozen_finetune_parameters(self):
        """Freeze block1 (temporal + spatial conv), train the rest — the
        base paper's same-session fine-tuning protocol."""
        for p in self.block1.parameters():
            p.requires_grad = False
        return [p for p in self.parameters() if p.requires_grad]
