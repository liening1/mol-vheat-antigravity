import torch
import torch.nn as nn
from .vheat import vHeat, LayerNorm2d

class MolVHeat(nn.Module):
    def __init__(self, task_type='regression', pretrained=False, **kwargs):
        super().__init__()
        self.task_type = task_type
        
        # Initialize vHeat backbone
        # We use default parameters from the official code or smaller ones if needed for speed
        # For now, defaults: patch_size=4, dims=[96, 192, 384, 768], depths=[2, 2, 9, 2]
        self.backbone = vHeat(**kwargs)
        
        # Remove original classifier to save memory/confusion (optional)
        del self.backbone.classifier
        
        self.num_features = self.backbone.num_features
        
        # Prediction Head
        # Global Average Pooling is done in the head
        self.head = nn.Sequential(
            LayerNorm2d(self.num_features),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(self.num_features, self.num_features // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.num_features // 2, 1)
        )
        
    def forward(self, x):
        # x: (B, 3, 224, 224)
        features = self.backbone.forward_features(x) # (B, C, H, W)
        out = self.head(features) # (B, 1)
        
        if self.task_type == 'classification':
            # Return logits for BCEWithLogitsLoss
            return out
        else:
            # Regression
            return out

if __name__ == '__main__':
    # Test model
    model = MolVHeat(task_type='regression')
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
