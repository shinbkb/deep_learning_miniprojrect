"""CIFAR-100 loaders and one validated stratified split shared by both models."""
import hashlib

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .config import project_path

CIFAR_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR_STD = (0.2675, 0.2565, 0.2761)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalization(config):
    return (CIFAR_MEAN, CIFAR_STD) if config["model"] == "CIFARResNet" else (IMAGENET_MEAN, IMAGENET_STD)


def make_transforms(config, train=False):
    ops = []
    if config["model"] == "swin_t":
        ops.append(transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC))
    if train:
        size = config["image_size"]
        ops += [transforms.RandomCrop(size, padding=4 if size == 32 else 12, padding_mode="reflect"),
                transforms.RandomHorizontalFlip()]
        if config["model"] == "CIFARResNet":
            ops.append(transforms.RandomRotation(15))
    return transforms.Compose(ops + [transforms.ToTensor(), transforms.Normalize(*normalization(config))])


def validate_split(train_idx, val_idx, size, validation_size):
    train_idx, val_idx = np.asarray(train_idx), np.asarray(val_idx)
    if train_idx.ndim != 1 or val_idx.ndim != 1:
        raise ValueError("Split indices must be one-dimensional")
    if train_idx.dtype.kind not in "iu" or val_idx.dtype.kind not in "iu":
        raise ValueError("Split indices must be integers")
    if len(train_idx) != size - validation_size or len(val_idx) != validation_size:
        raise ValueError("Split sizes do not match config")
    merged = np.concatenate([train_idx, val_idx])
    if not np.array_equal(np.sort(merged), np.arange(size)):
        raise ValueError("Split contains overlaps, duplicates, missing or out-of-range indices")


def get_split(targets, config):
    path = project_path(config["split_path"])
    fingerprint = hashlib.sha256(np.asarray(targets, dtype=np.int64).tobytes()).hexdigest()
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            if "seed" not in data or "labels_sha256" not in data:
                raise ValueError("Split lacks metadata. Use a new split_path for this pipeline.")
            if int(data["seed"]) != config["seed"] or str(data["labels_sha256"]) != fingerprint:
                raise ValueError("Existing split seed or dataset labels differ from this run")
            train_idx, val_idx = data["train_idx"], data["val_idx"]
    else:
        train_idx, val_idx = train_test_split(np.arange(len(targets)), test_size=config["validation_size"],
                                              random_state=config["seed"], stratify=targets)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation avoids silently replacing a split used by another experiment.
        with path.open("xb") as handle:
            np.savez(handle, train_idx=train_idx, val_idx=val_idx, seed=config["seed"], labels_sha256=fingerprint)
    validate_split(train_idx, val_idx, len(targets), config["validation_size"])
    return train_idx, val_idx


def seed_worker(_worker_id):
    import random
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def loader(dataset, config, device, shuffle=False):
    return DataLoader(dataset, batch_size=config["batch_size"], shuffle=shuffle,
                      num_workers=config["num_workers"], pin_memory=device.type == "cuda",
                      worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(config["seed"]))


def train_loaders(config, device, download=False):
    root = str(project_path(config["data_dir"]))
    augmented = datasets.CIFAR100(root, train=True, download=download, transform=make_transforms(config, True))
    clean = datasets.CIFAR100(root, train=True, download=False, transform=make_transforms(config))
    train_idx, val_idx = get_split(clean.targets, config)
    return (loader(Subset(augmented, train_idx), config, device, True),
            loader(Subset(clean, val_idx), config, device),
            loader(Subset(clean, train_idx), config, device),
            clean.classes, train_idx, val_idx)


def evaluation_loader(config, checkpoint, split, device, download=False):
    dataset = datasets.CIFAR100(str(project_path(config["data_dir"])), train=split != "test",
                               download=download, transform=make_transforms(config))
    if list(dataset.classes) != checkpoint["class_names"]:
        raise ValueError("Checkpoint label order differs from dataset")
    if split != "test":
        train_idx, val_idx = checkpoint["train_indices"], checkpoint["val_indices"]
        validate_split(train_idx, val_idx, len(dataset), config["validation_size"])
        dataset = Subset(dataset, train_idx if split == "train" else val_idx)
    return loader(dataset, config, device)
