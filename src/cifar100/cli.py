"""Command-line entry points. Heavy imports follow argument parsing."""
import argparse


def train_main():
    parser = argparse.ArgumentParser(description='Train custom ResNet + MixUp or pretrained Swin-T on CIFAR-100.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--download', action='store_true', help='Allow CIFAR-100 download; Swin weights also need downloading if not cached')
    parser.add_argument('--run-name')
    args = parser.parse_args()
    from .config import load_config
    from .utils import resolve_device
    from .pipeline import train
    train(load_config(args.config), resolve_device(args.device), args.download, args.run_name)


def evaluate_main():
    parser = argparse.ArgumentParser(description='Evaluate a Python-pipeline checkpoint and export plots/metrics.')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='val')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--data-dir')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    import torch
    from datetime import datetime
    from uuid import uuid4
    from .config import project_path
    from .checkpoint import load_checkpoint
    from .data import evaluation_loader
    from .engine import evaluate
    from .reporting import save_evaluation
    from .utils import resolve_device, seed_everything
    device = resolve_device(args.device)
    path = project_path(args.checkpoint)
    model, checkpoint = load_checkpoint(path, device)
    config = dict(checkpoint['config'])
    for key in ['data_dir', 'batch_size', 'num_workers']:
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if config['batch_size'] < 1 or config['num_workers'] < 0:
        parser.error('batch-size must be positive; num-workers must be non-negative')
    seed_everything(config['seed'])
    loader = evaluation_loader(config, checkpoint, args.split, device, args.download)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=config.get('label_smoothing', 0.))
    metrics = evaluate(model, loader, criterion, device, collect=True)
    output = project_path(config['results_dir']) / path.parent.name / 'evaluations' / (args.split + '_' + datetime.now().strftime('%Y%m%d_%H%M%S_') + uuid4().hex[:6])
    summary = save_evaluation(metrics, output, checkpoint['class_names'], config,
                              dict(split=args.split, checkpoint=str(path), model=config['model'], config=config))
    print(f"Accuracy: {summary['accuracy']:.2f}% | Macro F1: {summary['macro_f1']:.4f}\nResults: {output}")


def predict_main():
    parser = argparse.ArgumentParser(description='Predict CIFAR-100 labels for an image.')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    if not 1 <= args.top_k <= 100:
        parser.error('top-k must be between 1 and 100')
    import json
    import torch
    from PIL import Image
    from .config import project_path
    from .checkpoint import load_checkpoint
    from .data import make_transforms
    from .utils import resolve_device
    device = resolve_device(args.device)
    model, checkpoint = load_checkpoint(project_path(args.checkpoint), device)
    with Image.open(project_path(args.image)) as image:
        image = image.convert('RGB')
        if checkpoint['config']['model'] == 'CIFARResNet':
            image = image.resize((32, 32), Image.Resampling.BICUBIC)
        inputs = make_transforms(checkpoint['config'])(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        values, indices = model(inputs).softmax(1).topk(args.top_k)
    print(json.dumps([dict(label=checkpoint['class_names'][i], probability=p)
                      for p, i in zip(values[0].tolist(), indices[0].tolist())], indent=2))
