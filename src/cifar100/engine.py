"""Shared training and evaluation; accuracy is always expressed in percent."""
import numpy as np
import torch


def train_one_epoch(model, loader, criterion, optimizer, device, mixup_alpha=0., scaler=None):
    model.train()
    total_loss = correct = count = 0
    amp = scaler is not None and scaler.is_enabled()
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        lam = float(np.random.beta(mixup_alpha, mixup_alpha)) if mixup_alpha > 0 else 1.
        permutation = torch.randperm(len(labels), device=device) if mixup_alpha > 0 else torch.arange(len(labels), device=device)
        mixed = lam * inputs + (1 - lam) * inputs[permutation]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(mixed)
            loss = lam * criterion(logits, labels) + (1 - lam) * criterion(logits, labels[permutation])
        if amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        predictions = logits.argmax(1)
        correct += lam * predictions.eq(labels).sum().item() + (1 - lam) * predictions.eq(labels[permutation]).sum().item()
        total_loss += loss.item() * len(labels)
        count += len(labels)
    if not count:
        raise ValueError('Training loader is empty')
    return {'loss': total_loss / count, 'accuracy': 100 * correct / count}


@torch.no_grad()
def evaluate(model, loader, criterion, device, collect=False):
    model.eval()
    total_loss = correct = count = 0
    labels_all, predictions_all, confidence_all = [], [], []
    examples = []
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        logits = model(inputs)
        loss = criterion(logits, labels)
        confidence, predictions = logits.softmax(1).max(1)
        total_loss += loss.item() * len(labels)
        correct += predictions.eq(labels).sum().item()
        count += len(labels)
        if collect:
            labels_all.extend(labels.cpu().tolist())
            predictions_all.extend(predictions.cpu().tolist())
            confidence_all.extend(confidence.cpu().tolist())
            for i in range(min(len(labels), 16 - len(examples))):
                examples.append((inputs[i].cpu(), labels[i].item(), predictions[i].item()))
    if not count:
        raise ValueError('Evaluation loader is empty')
    metrics = {'loss': total_loss / count, 'accuracy': 100 * correct / count, 'samples': count}
    if collect:
        metrics.update(labels=labels_all, predictions=predictions_all, confidence=confidence_all, examples=examples)
    return metrics
