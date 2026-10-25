"""Training orchestration; the test split is only opened by evaluate.py."""
import logging
import time
import torch
from .checkpoint import load_checkpoint, save_checkpoint
from .data import train_loaders
from .engine import evaluate, train_one_epoch
from .models import build_model
from .models.swin import head_optimizer, finetune_optimizer
from .reporting import save_history
from .utils import create_run, seed_everything, save_json


def run_stage(model, loaders, criterion, optimizer, scheduler, device, config, classes,
              indices, stage, epochs, checkpoints, output, history, logger):
    train_loader, val_loader = loaders
    best_acc, best_loss = -1., float('inf')
    scaler = torch.amp.GradScaler('cuda', enabled=config.get('amp', False) and device.type == 'cuda')
    for stage_epoch in range(1, epochs + 1):
        started = time.perf_counter()
        learning_rate = optimizer.param_groups[0]['lr']
        train = train_one_epoch(model, train_loader, criterion, optimizer, device,
                                config.get('mixup_alpha', 0.), scaler)
        val = evaluate(model, val_loader, criterion, device)
        row = dict(epoch=len(history)+1, stage=stage, stage_epoch=stage_epoch,
                   train_loss=train['loss'], train_accuracy=train['accuracy'],
                   val_loss=val['loss'], val_accuracy=val['accuracy'], lr=learning_rate,
                   seconds=time.perf_counter()-started)
        history.append(row)
        for improved, name in [(val['accuracy'] > best_acc, 'best_accuracy'), (val['loss'] < best_loss, 'best_loss')]:
            if improved:
                save_checkpoint(checkpoints / f'{stage}_{name}.pt', model, config, classes,
                                *indices, stage, stage_epoch, val)
        best_acc, best_loss = max(best_acc, val['accuracy']), min(best_loss, val['loss'])
        if scheduler is not None:
            scheduler.step()
        save_history(history, output)
        logger.info('%s %d/%d | train loss %.4f acc %.2f%% | val loss %.4f acc %.2f%%',
                    stage, stage_epoch, epochs, train['loss'], train['accuracy'], val['loss'], val['accuracy'])
    return checkpoints / f'{stage}_best_accuracy.pt'


def train(config, device, download=False, run_name=None):
    seed_everything(config['seed'])
    output, checkpoints = create_run(config, run_name)
    logger = logging.getLogger('cifar100.' + output.name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handlers = [logging.StreamHandler(), logging.FileHandler(output / 'logs/train.log', encoding='utf-8')]
    for handler in handlers:
        logger.addHandler(handler)
    try:
        train_loader, val_loader, clean_loader, classes, train_idx, val_idx = train_loaders(config, device, download)
        model = build_model(config).to(device)
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=config.get('label_smoothing', 0.))
        history = []
        started = time.perf_counter()
        if config['model'] == 'CIFARResNet':
            optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'], weight_decay=config['weight_decay'])
            stages = [('train', config['epochs'], optimizer)]
        else:
            stages = [('head', config['head_epochs'], head_optimizer(model, config))]
        for stage, epochs, optimizer in stages:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs) if stage == 'train' else None
            best = run_stage(model, (train_loader, val_loader), criterion, optimizer, scheduler, device,
                             config, classes, (train_idx, val_idx), stage, epochs, checkpoints, output, history, logger)
        if config['model'] == 'swin_t':
            model, _ = load_checkpoint(best, device)
            optimizer = finetune_optimizer(model, config)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['finetune_epochs'])
            best = run_stage(model, (train_loader, val_loader), criterion, optimizer, scheduler, device,
                             config, classes, (train_idx, val_idx), 'finetune', config['finetune_epochs'],
                             checkpoints, output, history, logger)
        training_seconds = time.perf_counter() - started
        model, checkpoint = load_checkpoint(best, device)
        clean = evaluate(model, clean_loader, criterion, device)
        summary = dict(model=config['model'], best_checkpoint=str(best), selected_stage=checkpoint['stage'],
                       selected_epoch=checkpoint['epoch'], validation=checkpoint['validation'], clean_train=clean,
                       parameters=sum(p.numel() for p in model.parameters()), training_seconds=training_seconds,
                       torch_version=str(torch.__version__), device=str(device),
                       train_accuracy_definition='MixUp weighted' if config.get('mixup_alpha', 0) > 0 else 'standard')
        save_json(output / 'metrics/training_summary.json', summary)
        logger.info('Saved run: %s\nBest checkpoint: %s\nClean train accuracy: %.2f%%', output, best, clean['accuracy'])
        return summary
    finally:
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()
