import torch
import torch.nn as nn
from .vheat import vHeat, LayerNorm2d


class MolVHeat(nn.Module):
    """
    Mol-vHeat: vHeat backbone adapted for molecular property prediction.
    
    Args:
        task_type: 'regression' or 'classification'
        dropout: Dropout rate for the prediction head (default: 0.1 for regression, 0.3 for classification)
        **kwargs: Additional arguments passed to vHeat backbone
    """
    def __init__(self, task_type='regression', dropout=None, **kwargs):
        super().__init__()
        self.task_type = task_type
        
        # Set default dropout based on task type (classification needs more regularization)
        if dropout is None:
            dropout = 0.3 if task_type == 'classification' else 0.1
        
        # Initialize vHeat backbone
        self.backbone = vHeat(**kwargs)
        
        # Remove original classifier
        del self.backbone.classifier
        
        self.num_features = self.backbone.num_features
        
        # Prediction Head with configurable dropout
        self.head = nn.Sequential(
            LayerNorm2d(self.num_features),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Dropout(dropout),  # Dropout before first linear
            nn.Linear(self.num_features, self.num_features // 2),
            nn.ReLU(),
            nn.Dropout(dropout),  # Dropout after ReLU
            nn.Linear(self.num_features // 2, 1)
        )
        
    def forward(self, x):
        # x: (B, 3, 224, 224)
        features = self.backbone.forward_features(x)  # (B, C, H, W)
        out = self.head(features)  # (B, 1)
        return out


if __name__ == '__main__':
    # Test model
    model = MolVHeat(task_type='classification')
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
