"""Seeding, device selection and isolated output directories."""
import json
import random
import re
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch

from .config import project_path


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --device cpu or a compatible CUDA environment.")
    return torch.device(name)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def create_run(config, name=None):
    name = name or datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:6]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError("Run name may contain letters, numbers, _, . and - only")
    result = project_path(config["results_dir"]) / name
    checkpoint = project_path(config["checkpoint_dir"]) / name
    if result.exists() or checkpoint.exists():
        raise FileExistsError(f"Run {name!r} already exists. Choose another --run-name.")
    result.mkdir(parents=True)
    checkpoint.mkdir(parents=True)
    for child in ("metrics", "figures", "logs"):
        (result / child).mkdir()
    save_json(result / "config.json", config)
    return result, checkpoint
