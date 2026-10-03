import json
import os

import torch
import torchvision.models as models
from torch import nn


class _PretrainedMixin:
    """save/load helpers shared by the condition encoders (diffusers-style folder layout)."""

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(self.config, f)

    def register_to_config(self, **kwargs):
        if not hasattr(self, "config"):
            self.config = {}
        self.config.update(kwargs)

    @classmethod
    def from_pretrained(cls, load_directory, subfolder=None):
        directory = load_directory if subfolder is None else os.path.join(load_directory, subfolder)
        with open(os.path.join(directory, "config.json"), "r") as f:
            config = json.load(f)
        state_dict = torch.load(os.path.join(directory, "pytorch_model.bin"), map_location="cpu")
        config.pop("num_tokens", None)  # written by older checkpoints
        model = cls(**config)
        model.load_state_dict(state_dict)
        return model


def _mlp(input_dim, output_dim, hidden_dims):
    """Linear stack input_dim -> hidden_dims... -> output_dim as a ModuleList."""
    layers = nn.ModuleList()
    for h in hidden_dims:
        layers.append(nn.Linear(input_dim, h))
        input_dim = h
    layers.append(nn.Linear(input_dim, output_dim))
    return layers


class condition_encoder(nn.Module, _PretrainedMixin):
    """One-hot condition (B, 1, 10) -> (B, 1, output_dim). Used for count and position runs."""

    def __init__(self, input_dim=10, output_dim=64, dropout_rate=0.1, hidden_dims=None):
        super().__init__()
        self.config = {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_dims": hidden_dims,
            "dropout_rate": dropout_rate,
        }
        self.hidden_dims = hidden_dims
        # Parameter names are kept for checkpoint compatibility: `fc` is a single Linear without hidden layers.
        self.fc = _mlp(input_dim, output_dim, hidden_dims) if hidden_dims else nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)

    @property
    def dtype(self):
        layer = self.fc[0] if isinstance(self.fc, nn.ModuleList) else self.fc
        return layer.weight.dtype

    def forward(self, x):
        layers = self.fc if isinstance(self.fc, nn.ModuleList) else [self.fc]
        for layer in layers:
            x = self.dropout(self.relu(layer(x)))
        return x


class two_condition_encoder(nn.Module, _PretrainedMixin):
    """Two one-hot tokens (B, 2, 10) -> (B, 2, output_dim), one MLP per token.
    Used for attribute (color1, color2) and composition (color, position) runs.
    Parameter names (`fc_color`, `fc_counting`) are historical and kept for checkpoint compatibility."""

    def __init__(self, input_dim=10, output_dim=64, dropout_rate=0.1, hidden_dims=None):
        super().__init__()
        self.config = {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_dims": hidden_dims,
            "dropout_rate": dropout_rate,
        }
        self.hidden_dims = hidden_dims
        self.use_hidden = bool(hidden_dims)
        if self.use_hidden:
            self.fc_color = _mlp(input_dim, output_dim, hidden_dims)
            self.fc_counting = _mlp(input_dim, output_dim, hidden_dims)
        else:
            self.fc_color = nn.Linear(input_dim, output_dim)
            self.fc_counting = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)

    @property
    def dtype(self):
        layer = self.fc_color[0] if self.use_hidden else self.fc_color
        return layer.weight.dtype

    def _run(self, layers, x):
        for layer in (layers if self.use_hidden else [layers]):
            x = self.dropout(self.relu(layer(x)))
        return x

    def forward(self, x):
        first = self._run(self.fc_color, x[:, 0, :])
        second = self._run(self.fc_counting, x[:, 1, :])
        if self.use_hidden:
            # the original hidden-layer variant applied relu/dropout once more after the last layer; keep it
            first = self.dropout(self.relu(first))
            second = self.dropout(self.relu(second))
        return torch.stack((first, second), dim=1)  # (B, 2, C)


class Classifier(nn.Module):
    """CNN count classifier (classifier_weights/best_classifier_count.pth)."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 128x128 -> 64x64
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            # Block 2: 64x64 -> 32x32
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            # Block 3: 32x32 -> 16x16
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3),
            # Block 4: 16x16 -> 8x8
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class PretrainedClassifier(nn.Module):
    """ResNet18 classifier (position / attribute / shape / colour weights in classifier_weights/)."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)
