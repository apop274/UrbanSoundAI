"""
Simple CNN for classifying mel spectrograms (single channel 2D input).
"""
import torch
import torch.nn as nn

from . import config


class SpectrogramCNN(nn.Module):
    """
    Small CNN: input (1, n_mels, time_steps).
    Conv blocks + global pool + FC -> num_classes.
    """

    def __init__(self, num_classes=None):
        super().__init__()
        num_classes = num_classes or config.NUM_CLASSES
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
        )
        # After 3 pool(2): 128/8=16 mel, 173/8≈21 time (floor)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


def get_model(num_classes=None):
    """Return model and class list for inference."""
    return SpectrogramCNN(num_classes=num_classes or config.NUM_CLASSES)
