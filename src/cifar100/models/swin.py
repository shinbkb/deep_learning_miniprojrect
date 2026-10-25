"""Swin-T model and the two optimizer stages from the transfer learning notebook."""
import torch
from torch import nn
from torchvision.models import swin_t, Swin_T_Weights


def build_swin(pretrained=True):
    model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1 if pretrained else None)
    model.requires_grad_(False)
    model.head = nn.Linear(model.head.in_features, 100)
    return model


def head_optimizer(model, config):
    model.requires_grad_(False)
    model.head.requires_grad_(True)
    return torch.optim.AdamW(model.head.parameters(), lr=config["head_lr"],
                             weight_decay=config["weight_decay"])


def finetune_optimizer(model, config):
    model.requires_grad_(True)
    return torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if not n.startswith("head.")],
         "lr": config["backbone_lr"]},
        {"params": model.head.parameters(), "lr": config["finetune_head_lr"]},
    ], weight_decay=config["weight_decay"])
