"""Export numerical results and report-ready figures without a display server."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from .data import normalization
from .utils import save_json


def write_csv(path, rows, fields):
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_history(rows, output):
    output = Path(output)
    write_csv(output / 'metrics/history.csv', rows, list(rows[0]))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, ('loss', 'accuracy')):
        for source in ('train', 'val'):
            label = 'Train (MixUp weighted if enabled)' if source == 'train' else 'Validation (clean)'
            ax.plot([r['epoch'] for r in rows], [r[f'{source}_{metric}'] for r in rows], label=label)
        ax.set(xlabel='Epoch', ylabel=metric, title=metric.capitalize())
        ax.legend(fontsize=8)
        ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(output / 'figures/learning_curves.png', dpi=180)
    plt.close(fig)


def save_evaluation(metrics, output, classes, config, metadata):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    truth, pred = metrics['labels'], metrics['predictions']
    ids = list(range(len(classes)))
    report = classification_report(truth, pred, labels=ids, target_names=classes, output_dict=True, zero_division=0)
    summary = {k: metrics[k] for k in ('loss', 'accuracy', 'samples')}
    summary.update(macro_f1=float(f1_score(truth, pred, labels=ids, average='macro', zero_division=0)), **metadata)
    save_json(output / 'summary.json', summary)
    save_json(output / 'classification_report.json', report)
    write_csv(output / 'per_class_metrics.csv', [{'class': name, **report[name]} for name in classes],
              ['class', 'precision', 'recall', 'f1-score', 'support'])
    write_csv(output / 'predictions.csv', [dict(index=i, true=t, predicted=p, confidence=c)
              for i, (t, p, c) in enumerate(zip(truth, pred, metrics['confidence']))], ['index', 'true', 'predicted', 'confidence'])
    cm = confusion_matrix(truth, pred, labels=ids)
    normalized = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    for name, matrix in [('confusion_counts', cm), ('confusion_normalized', normalized)]:
        np.savetxt(output / f'{name}.csv', matrix, delimiter=',', header=','.join(classes), comments='')
        fig, ax = plt.subplots(figsize=(16, 14))
        im = ax.imshow(matrix, cmap='Blues')
        ax.set(xticks=ids, yticks=ids, xticklabels=classes, yticklabels=classes,
               xlabel='Predicted class', ylabel='True class', title=name.replace('_', ' ').title())
        ax.tick_params(axis='x', labelrotation=90, labelsize=5)
        ax.tick_params(axis='y', labelsize=5)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(output / f'{name}.png', dpi=180)
        plt.close(fig)
    errors = [(int(cm[i,j]), classes[i], classes[j]) for i in ids for j in ids if i != j and cm[i,j] > 0]
    errors = sorted(errors, reverse=True)[:15]
    if errors:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh([f'{t} → {p}' for _,t,p in errors][::-1], [n for n,_,_ in errors][::-1])
        ax.set(xlabel='Number of errors', title='Most frequent confusions')
        fig.tight_layout()
        fig.savefig(output / 'top_confusions.png', dpi=180)
        plt.close(fig)
    if metrics['examples']:
        mean, std = map(np.asarray, normalization(config))
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        for ax in axes.flat:
            ax.axis('off')
        for ax, (tensor, target, prediction) in zip(axes.flat, metrics['examples']):
            image = np.clip(tensor.permute(1,2,0).numpy() * std + mean, 0, 1)
            ax.imshow(image)
            ax.set_title(f'True: {classes[target]}\nPred: {classes[prediction]}', fontsize=9,
                         color='green' if target == prediction else 'red')
        fig.tight_layout()
        fig.savefig(output / 'sample_predictions.png', dpi=160)
        plt.close(fig)
    return summary
