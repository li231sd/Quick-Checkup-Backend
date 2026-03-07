import os
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights


class ABCDEModel(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        in_features = backbone.classifier[1].in_features
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.dropout = nn.Dropout(p=dropout)
        self.head_A = nn.Linear(in_features, 1)
        self.head_B = nn.Linear(in_features, 1)
        self.head_C = nn.Linear(in_features, 1)
        self.head_D = nn.Linear(in_features, 1)
        self.head_E = nn.Linear(in_features, 1)

    def forward(self, x):
        x = self.backbone(x)
        x = x.flatten(1)
        x = self.dropout(x)
        A = torch.sigmoid(self.head_A(x))
        B = torch.sigmoid(self.head_B(x))
        C = torch.sigmoid(self.head_C(x))
        D = torch.sigmoid(self.head_D(x))
        E = torch.sigmoid(self.head_E(x))
        return torch.cat([A, B, C, D, E], dim=1)


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')