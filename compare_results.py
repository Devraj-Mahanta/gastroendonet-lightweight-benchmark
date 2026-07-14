import json
import os
import glob
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import config as cfg
from models import MODEL_REGISTRY, MODEL_DISPLAY_NAMES
METRICS_TO_REPORT = ['accuracy', 'auc_macro', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro', 'inference_ms_per_image', 'model_size_mb', 'n_params']
METRIC_LABELS = {'accuracy': 'Accuracy', 'auc_macro': 'AUC (macro)', 'f1_macro': 'F1 (macro)', 'f1_weighted': 'F1 (weighted)', 'precision_macro': 'Precision', 'recall_macro': 'Recall', 'inference_ms_per_image': 'Inference (ms/img)', 'model_size_mb': 'Size (MB)', 'n_params': 'Parameters'}

def load_all_results():
    data = {m: [] for m in MODEL_REGISTRY}
    pattern = os.path.join(cfg.RESULTS_DIR, '*_run*_eval.json')
    for fpath in sorted(glob.glob(pattern)):
        with open(fpath) as f:
            rec = json.load(f)
        model_name = rec.get('model')
        if model_name in data:
            data[model_name].append(rec)
    return data

def aggregate(runs: list, metric: str):
    values = [r[metric] for r in runs if metric in r and r[metric] is not None]
    if not values:
        return (float('nan'), float('nan'))
    return (float(np.mean(values)), float(np.std(values)))

def print_table(data):
    header_metrics = ['accuracy', 'auc_macro', 'f1_macro', 'precision_macro', 'recall_macro', 'inference_ms_per_image', 'model_size_mb']
    col_w = 22
    head = f"{'Model':<35}" + ''.join((f'{METRIC_LABELS[m]:>{col_w}}' for m in header_metrics))
    print('\n' + '=' * len(head))
    print(' MODEL COMPARISON  (mean ± std over 3 runs)')
    print('=' * len(head))
    print(head)
    print('-' * len(head))
    for model_name in MODEL_REGISTRY:
        runs = data[model_name]
        if not runs:
            print(f'{MODEL_DISPLAY_NAMES[model_name]:<35}  (no results)')
            continue
        row = f'{MODEL_DISPLAY_NAMES[model_name]:<35}'
        for m in header_metrics:
            (mean, std) = aggregate(runs, m)
            if m in ('inference_ms_per_image', 'model_size_mb', 'n_params'):
                cell = f'{mean:.2f}' if not np.isnan(mean) else '—'
            else:
                cell = f'{mean:.4f}±{std:.4f}' if not np.isnan(mean) else '—'
            row += f'{cell:>{col_w}}'
        print(row)
    print('=' * len(head))

def save_csv(data):
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(cfg.RESULTS_DIR, 'comparison_table.csv')
    all_metrics = METRICS_TO_REPORT
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['model', 'display_name', 'n_runs']
        for m in all_metrics:
            header += [f'{m}_mean', f'{m}_std']
        writer.writerow(header)
        for model_name in MODEL_REGISTRY:
            runs = data[model_name]
            row = [model_name, MODEL_DISPLAY_NAMES[model_name], len(runs)]
            for m in all_metrics:
                (mean, std) = aggregate(runs, m)
                row += [f'{mean:.6f}', f'{std:.6f}']
            writer.writerow(row)
    print(f'\nCSV saved -> {csv_path}')

def save_bar_chart(data):
    metrics_plot = ['accuracy', 'f1_macro', 'auc_macro']
    labels_plot = ['Accuracy', 'F1 (macro)', 'AUC (macro)']
    model_names = list(MODEL_REGISTRY.keys())
    display = [MODEL_DISPLAY_NAMES[m].replace(' (pretrained)', '\n(pretrained)').replace(' (scratch)', '\n(scratch)') for m in model_names]
    (fig, axes) = plt.subplots(1, len(metrics_plot), figsize=(16, 5))
    fig.suptitle('Model Comparison  (mean ± std, 3 runs)', fontsize=13)
    for (ax, metric, label) in zip(axes, metrics_plot, labels_plot):
        (means, stds) = ([], [])
        for m in model_names:
            (mn, sd) = aggregate(data[m], metric)
            means.append(mn)
            stds.append(sd)
        x = np.arange(len(model_names))
        clr = ['#4C72B0' if 'pretrained' in MODEL_DISPLAY_NAMES[m] else '#DD8452' for m in model_names]
        bars = ax.bar(x, means, yerr=stds, capsize=4, color=clr, error_kw={'elinewidth': 1.5})
        ax.set_xticks(x)
        ax.set_xticklabels(display, fontsize=7, rotation=15, ha='right')
        ax.set_title(label)
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', alpha=0.3)
        for (bar, mn) in zip(bars, means):
            if not np.isnan(mn):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{mn:.3f}', ha='center', va='bottom', fontsize=7)
    from matplotlib.patches import Patch
    legend_elements = [Patch(color='#4C72B0', label='Pretrained backbone'), Patch(color='#DD8452', label='Trained from scratch')]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out_path = os.path.join(cfg.RESULTS_DIR, 'comparison_plots.png')
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'Chart  saved -> {out_path}')

def main():
    data = load_all_results()
    print_table(data)
    save_csv(data)
    save_bar_chart(data)
if __name__ == '__main__':
    main()
