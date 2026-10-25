"""Load runnable JSON configurations; all relative paths use the project root."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_config(path):
    config = json.loads(project_path(path).read_text(encoding="utf-8"))
    if config.get("model") not in ("CIFARResNet", "swin_t"):
        raise ValueError("model must be CIFARResNet or swin_t")
    for key in ("seed", "batch_size", "image_size", "validation_size", "num_workers"):
        if not isinstance(config[key], int):
            raise ValueError(f"{key} must be an integer")
    if config["batch_size"] <= 0 or not 0 < config["validation_size"] < 50000:
        raise ValueError("Invalid batch_size or validation_size")
    if config["num_workers"] < 0 or config["weight_decay"] < 0:
        raise ValueError("num_workers and weight_decay must be non-negative")
    if config["model"] == "CIFARResNet":
        if config["epochs"] < 1 or config["lr"] <= 0 or config["image_size"] != 32:
            raise ValueError("ResNet requires epochs >= 1, lr > 0 and image_size=32")
        if config["mixup_alpha"] < 0 or not 0 <= config["label_smoothing"] < 1:
            raise ValueError("Invalid MixUp / smoothing configuration")
        if not 0 <= config["dropout"] < 1:
            raise ValueError("dropout must be in [0,1)")
        if config["optimizer"] != "SGD" or config["scheduler"] != "CosineAnnealingLR":
            raise ValueError("ResNet pipeline supports SGD + CosineAnnealingLR")
    else:
        if min(config["head_epochs"], config["finetune_epochs"]) < 1:
            raise ValueError("Both Swin stages require at least one epoch")
        if min(config["head_lr"], config["backbone_lr"], config["finetune_head_lr"]) <= 0:
            raise ValueError("Learning rates must be positive")
        if config["pretrained"] != "IMAGENET1K_V1" or config["image_size"] != 224:
            raise ValueError("Swin pipeline uses IMAGENET1K_V1 and 224px input")
        if config["optimizer"] != "AdamW":
            raise ValueError("Swin pipeline supports AdamW")
    return config
