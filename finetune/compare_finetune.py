import sys, os
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)
import argparse, csv, glob, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import config as base_cfg
from finetune.config_finetune import ABLATION_CONFIGS, RESULTS_DIR, NUM_RUNS
from finetune.models import FT_REGISTRY, FT_DISPLAY_NAMES

def _agg(runs, key):
    vals = [r[key] for r in runs if key in r and r[key] is not None]
    if not vals:
        return (float('nan'), float('nan'))
    return (float(np.mean(vals)), float(np.std(vals)))

def _load_baseline(model_name):
    runs = []
    for r in range(1, NUM_RUNS + 1):
        path = os.path.join('results', f'{model_name}_run{r}_eval.json')
        if os.path.exists(path):
            with open(path) as f:
                runs.append(json.load(f))
    return runs

def _load_finetune(model_name, cfg_name):
    runs = []
    for r in range(1, NUM_RUNS + 1):
        path = os.path.join(RESULTS_DIR, f'{model_name}_{cfg_name}_run{r}_eval.json')
        if os.path.exists(path):
            with open(path) as f:
                runs.append(json.load(f))
    return runs
METRICS = ['accuracy', 'auc_macro', 'f1_macro', 'precision_macro', 'recall_macro']
LABELS = {'accuracy': 'Accuracy', 'auc_macro': 'AUC', 'f1_macro': 'F1 (macro)', 'precision_macro': 'Precision', 'recall_macro': 'Recall'}

def build_ablation_table(model_name):
    rows = []
    base_runs = _load_baseline(model_name)
    row = {'config': 'baseline', 'n_runs': len(base_runs), 'description': 'Original three-phase fine-tuning (no changes)'}
    for m in METRICS:
        (mean, std) = _agg(base_runs, m)
        row[f'{m}_mean'] = mean
        row[f'{m}_std'] = std
    rows.append(row)
    for (cfg_name, ft_cfg) in ABLATION_CONFIGS.items():
        if cfg_name == 'baseline':
            continue
        ft_runs = _load_finetune(model_name, cfg_name)
        if not ft_runs:
            continue
        row = {'config': cfg_name, 'n_runs': len(ft_runs), 'description': ft_cfg['description']}
        for m in METRICS:
            (mean, std) = _agg(ft_runs, m)
            row[f'{m}_mean'] = mean
            row[f'{m}_std'] = std
        rows.append(row)
    return rows

def print_ablation_table(model_name, rows):
    col_w = 14
    hdr = f"{'Config':<22}  " + ''.join((f'{LABELS[m]:>{col_w}}' for m in METRICS))
    sep = '-' * len(hdr)
    lines = []
    lines.append('')
    lines.append('=' * len(hdr))
    lines.append(f'  ABLATION TABLE — {FT_DISPLAY_NAMES.get(model_name, model_name)}')
    lines.append('=' * len(hdr))
    lines.append(hdr)
    lines.append(sep)
    for row in rows:
        cfg = row['config']
        cell = f'{cfg:<22}'
        for m in METRICS:
            (mn, sd) = (row.get(f'{m}_mean', float('nan')), row.get(f'{m}_std', float('nan')))
            if np.isnan(mn):
                cell += f"{'—':>{col_w}}"
            else:
                val = f'{mn:.4f}±{sd:.4f}'
                cell += f'{val:>{col_w}}'
        lines.append(cell)
    lines.append('=' * len(hdr))
    text = '\n'.join(lines)
    print(text)
    return text

def save_csv(model_name, rows):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f'ablation_table_{model_name}.csv')
    if not rows:
        return path
    fieldnames = ['config', 'n_runs', 'description'] + [f'{m}_{s}' for m in METRICS for s in ['mean', 'std']]
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print(f'CSV  saved -> {path}')
    return path

def save_txt(model_name, text):
    path = os.path.join(RESULTS_DIR, f'ablation_table_{model_name}.txt')
    with open(path, 'w') as f:
        f.write(text)
    print(f'Text saved -> {path}')

def save_plots(model_name, rows):
    if not rows:
        return
    cfg_labels = [r['config'] for r in rows]
    metrics_plot = ['accuracy', 'auc_macro', 'f1_macro']
    labels_plot = ['Accuracy', 'AUC (macro)', 'F1 (macro)']
    (fig, axes) = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'Ablation study — {FT_DISPLAY_NAMES.get(model_name, model_name)}', fontsize=13)
    colours = ['#4C72B0'] + ['#DD8452'] * (len(rows) - 1)
    for (ax, metric, label) in zip(axes, metrics_plot, labels_plot):
        means = [r.get(f'{metric}_mean', float('nan')) for r in rows]
        stds = [r.get(f'{metric}_std', 0.0) for r in rows]
        x = np.arange(len(rows))
        ax.bar(x, means, yerr=stds, capsize=4, color=colours, error_kw={'elinewidth': 1.5})
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace('_', '\n') for c in cfg_labels], fontsize=7, rotation=0, ha='center')
        ax.set_title(label)
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', alpha=0.3)
        for (xi, (mn, sd)) in enumerate(zip(means, stds)):
            if not np.isnan(mn):
                ax.text(xi, mn + sd + 0.01, f'{mn:.3f}', ha='center', va='bottom', fontsize=7)
    from matplotlib.patches import Patch
    legend = [Patch(color='#4C72B0', label='Original baseline'), Patch(color='#DD8452', label='Fine-tuned')]
    fig.legend(handles=legend, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out = os.path.join(RESULTS_DIR, f'ablation_plots_{model_name}.png')
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'Plot saved -> {out}')

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='efficientnet_b1', choices=list(FT_REGISTRY.keys()) + ['all'])
    return p.parse_args()

def main():
    args = parse_args()
    models = list(FT_REGISTRY.keys()) if args.model == 'all' else [args.model]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for model_name in models:
        rows = build_ablation_table(model_name)
        if not rows:
            print(f'[{model_name}] No results found. Run ablation_runner.py first.')
            continue
        text = print_ablation_table(model_name, rows)
        save_txt(model_name, text)
        save_csv(model_name, rows)
        save_plots(model_name, rows)
if __name__ == '__main__':
    main()
