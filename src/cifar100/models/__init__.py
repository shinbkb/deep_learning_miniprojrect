from .resnet import CIFARResNet
from .swin import build_swin


def build_model(config, pretrained=True):
    if config["model"] == "CIFARResNet":
        return CIFARResNet(dropout=config["dropout"])
    if config["model"] == "swin_t":
        return build_swin(pretrained=pretrained)
    raise ValueError(f"Unsupported model: {config['model']}")
