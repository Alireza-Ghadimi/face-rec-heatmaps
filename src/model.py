"""Model builder with ResNet backbone and ArcFace/CosFace heads."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


def build_backbone(name: str = "resnet50", pretrained: bool = True) -> nn.Module:
    if name != "resnet50":
        raise ValueError(f"Unsupported backbone: {name}")
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    try:
        model = resnet50(weights=weights)
    except Exception:
        model = resnet50(weights=None)
    return model


def patch_first_conv(model: nn.Module, in_ch: int) -> None:
    """Expand first conv to extra channels with mean init."""
    conv1: nn.Conv2d = model.conv1
    if conv1.in_channels == in_ch:
        return
    old_weight = conv1.weight.data
    new_weight = old_weight.mean(dim=1, keepdim=True).repeat(1, in_ch, 1, 1)
    new_weight[:, : old_weight.shape[1]] = old_weight
    model.conv1 = nn.Conv2d(
        in_ch,
        conv1.out_channels,
        kernel_size=conv1.kernel_size,
        stride=conv1.stride,
        padding=conv1.padding,
        bias=conv1.bias is not None,
    )
    model.conv1.weight.data = new_weight


class ArcFaceHead(nn.Module):
    def __init__(self, feat_dim: int, num_classes: int, margin: float = 0.5, scale: float = 64.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.weight)
        self.margin = margin
        self.scale = scale

    def forward(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        feats_norm = F.normalize(feats)
        w_norm = F.normalize(self.weight)
        cosine = F.linear(feats_norm, w_norm)
        one_hot = F.one_hot(labels, num_classes=w_norm.size(0)).float()
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        logits = torch.cos(theta + self.margin)
        logits = self.scale * torch.where(one_hot.bool(), logits, cosine)
        return logits


class CosFaceHead(nn.Module):
    def __init__(self, feat_dim: int, num_classes: int, margin: float = 0.35, scale: float = 64.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.weight)
        self.margin = margin
        self.scale = scale

    def forward(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        feats_norm = F.normalize(feats)
        w_norm = F.normalize(self.weight)
        cosine = F.linear(feats_norm, w_norm)
        one_hot = F.one_hot(labels, num_classes=w_norm.size(0)).float()
        logits = cosine - one_hot * self.margin
        return logits * self.scale


class FaceModel(nn.Module):
    """Backbone + margin head; returns logits during training and embeddings otherwise."""

    def __init__(
        self,
        num_classes: int,
        in_ch: int,
        backbone: str = "resnet50",
        pretrained: bool = True,
        use_cosface: bool = False,
        arc_margin: float = 0.5,
        arc_scale: float = 64.0,
        cos_margin: float = 0.35,
        cos_scale: float = 64.0,
    ) -> None:
        super().__init__()
        self.backbone = build_backbone(backbone, pretrained)
        patch_first_conv(self.backbone, in_ch)
        feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        if use_cosface:
            self.head = CosFaceHead(feat_dim, num_classes, margin=cos_margin, scale=cos_scale)
        else:
            self.head = ArcFaceHead(feat_dim, num_classes, margin=arc_margin, scale=arc_scale)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        feats = self.encode(x)
        if labels is None:
            return feats
        return self.head(feats, labels)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return F.normalize(feats)
