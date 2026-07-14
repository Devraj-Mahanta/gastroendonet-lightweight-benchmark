import sys, os
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)
import argparse, csv, glob, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import config as base_cfg
from models import MODEL_REGISTRY, MODEL_DISPLAY_NAMES
from finetune.config_finetune import RESULTS_DIR
METRICS = ['accuracy', 'auc_macro', 'f1_macro', 'precision_macro', 'recall_macro', 'inference_ms_per_image', 'model_size_mb']
LABELS = {'accuracy': 'Accuracy', 'auc_macro': 'AUC (macro)', 'f1_macro': 'F1 (macro)', 'precision_macro': 'Precision', 'recall_macro': 'Recall', 'inference_ms_per_image': 'Inference (ms/img)', 'model_size_mb': 'Size (MB)'}
FT_DISPLAY = {'efficientnet_b1': 'EfficientNet-B1 (fine-tuned [FT])', 'mobilenetv3_small': 'MobileNetV3-Small (fine-tuned [FT])'}

def _agg(runs, key):
    vals = [r[key] for r in runs if key in r and r[key] is not None]
    if not vals:
        return (float('nan'), float('nan'))
    return (float(np.mean(vals)), float(np.std(vals)))

def load_original_results():
    data = {m: [] for m in MODEL_REGISTRY}
    for fpath in sorted(glob.glob(os.path.join('results', '*_run*_eval.json'))):
        with open(fpath) as f:
            rec = json.load(f)
        m = rec.get('model')
        if m in data:
            data[m].append(rec)
    return data

def load_finetune_results(config_name='full'):
    from finetune.models import FT_REGISTRY
    data = {}
    for model_name in FT_REGISTRY:
        runs = []
        for r in range(1, 4):
            path = os.path.join(RESULTS_DIR, f'{model_name}_{config_name}_run{r}_eval.json')
            if os.path.exists(path):
                with open(path) as f:
                    runs.append(json.load(f))
        if runs:
            data[model_name] = runs
    return data

def build_rows(orig_data, ft_data):
    rows = []
    for model_name in MODEL_REGISTRY:
        runs = orig_data.get(model_name, [])
        row = {'model': model_name, 'display_name': MODEL_DISPLAY_NAMES[model_name], 'n_runs': len(runs), 'group': 'original'}
        for m in METRICS:
            (mean, std) = _agg(runs, m)
            row[f'{m}_mean'] = mean
            row[f'{m}_std'] = std
        rows.append(row)
    for (model_name, runs) in ft_data.items():
        row = {'model': model_name + '_ft', 'display_name': FT_DISPLAY.get(model_name, model_name + ' (fine-tuned)'), 'n_runs': len(runs), 'group': 'finetuned'}
        for m in METRICS:
            (mean, std) = _agg(runs, m)
            row[f'{m}_mean'] = mean
            row[f'{m}_std'] = std
        rows.append(row)
    return rows

def print_table(rows):
    plot_metrics = ['accuracy', 'auc_macro', 'f1_macro', 'precision_macro', 'recall_macro', 'inference_ms_per_image', 'model_size_mb']
    col_w = 20
    head = f"{'Model':<42}" + ''.join((f'{LABELS[m]:>{col_w}}' for m in plot_metrics))
    sep = '=' * len(head)
    dash = '-' * len(head)
    lines = ['', sep, ' ALL-MODEL COMPARISON  (mean ± std over 3 runs)', sep, head, dash]
    prev_group = None
    for row in rows:
        if prev_group and row['group'] != prev_group:
            lines.append(dash)
            lines.append(f'  Fine-tuned models')
            lines.append(dash)
        prev_group = row['group']
        name = row['display_name']
        if not row['n_runs']:
            lines.append(f'{name:<42}  (no results)')
            continue
        line = f'{name:<42}'
        for m in plot_metrics:
            mn = row.get(f'{m}_mean', float('nan'))
            sd = row.get(f'{m}_std', float('nan'))
            if np.isnan(mn):
                cell = '—'
            elif m in ('inference_ms_per_image', 'model_size_mb'):
                cell = f'{mn:.2f}'
            else:
                cell = f'{mn:.4f}±{sd:.4f}'
            line += f'{cell:>{col_w}}'
        lines.append(line)
    lines.append(sep)
    text = '\n'.join(lines)
    print(text)
    return text

def save_csv(rows):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, 'final_comparison.csv')
    fields = ['model', 'display_name', 'group', 'n_runs'] + [f'{m}_{s}' for m in METRICS for s in ['mean', 'std']]
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print(f'\nCSV  -> {path}')
    return path

def save_txt(text):
    path = os.path.join(RESULTS_DIR, 'final_comparison.txt')
    with open(path, 'w') as f:
        f.write(text)
    print(f'Text -> {path}')

def save_bar_chart(rows):
    metrics_plot = ['accuracy', 'f1_macro', 'auc_macro']
    labels_plot = ['Accuracy', 'F1 (macro)', 'AUC (macro)']

    def _colour(row):
        if row['group'] == 'finetuned':
            return '#2ca02c'
        if 'pretrained' in row['display_name']:
            return '#4C72B0'
        return '#DD8452'
    display = [r['display_name'].replace(' (pretrained)', '\n(pretrained)').replace(' (scratch)', '\n(scratch)').replace(' (fine-tuned [FT])', '\n(fine-tuned [FT])') for r in rows]
    colours = [_colour(r) for r in rows]
    (fig, axes) = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('All-Model Comparison  (mean ± std, 3 runs)', fontsize=13)
    for (ax, metric, label) in zip(axes, metrics_plot, labels_plot):
        means = [r.get(f'{metric}_mean', float('nan')) for r in rows]
        stds = [r.get(f'{metric}_std', 0.0) for r in rows]
        x = np.arange(len(rows))
        bars = ax.bar(x, means, yerr=stds, capsize=4, color=colours, error_kw={'elinewidth': 1.5}, edgecolor=['black' if r['group'] == 'finetuned' else 'none' for r in rows], linewidth=1.5)
        ax.set_xticks(x)
        ax.set_xticklabels(display, fontsize=6.5, rotation=15, ha='right')
        ax.set_title(label, fontsize=11)
        ax.set_ylim(0, 1.08)
        ax.grid(axis='y', alpha=0.3)
        for (bar, mn, sd) in zip(bars, means, stds):
            if not np.isnan(mn):
                ax.text(bar.get_x() + bar.get_width() / 2, mn + sd + 0.012, f'{mn:.3f}', ha='center', va='bottom', fontsize=6.5, fontweight='bold' if bar.get_edgecolor()[0] < 1 else 'normal')
    legend = [mpatches.Patch(color='#4C72B0', label='Pretrained (original)'), mpatches.Patch(color='#DD8452', label='From scratch (original)'), mpatches.Patch(color='#2ca02c', label='Fine-tuned (this work)', linewidth=1.5, edgecolor='black')]
    fig.legend(handles=legend, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.01), fontsize=9)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out = os.path.join(RESULTS_DIR, 'final_comparison_bar.png')
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'Bar  -> {out}')

def save_radar_chart(rows):
    radar_metrics = ['accuracy', 'auc_macro', 'f1_macro', 'precision_macro', 'recall_macro']
    radar_labels = ['Accuracy', 'AUC', 'F1', 'Precision', 'Recall']
    N = len(radar_metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    highlight = {'efficientnet_b1': ('#4C72B0', 'EfficientNet-B1 (original)'), 'mobilenetv3_small': ('#9467bd', 'MobileNetV3-Small (original)'), 'shufflenetv2': ('#DD8452', 'ShuffleNetV2 (scratch)')}
    (fig, ax) = plt.subplots(1, 1, figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels, size=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], size=7)
    ax.grid(alpha=0.3)
    plotted = []
    for row in rows:
        vals = [row.get(f'{m}_mean', 0) for m in radar_metrics]
        vals += vals[:1]
        name = row['display_name']
        if row['group'] == 'finetuned':
            colour = '#2ca02c'
            (lw, ls, alpha) = (2.5, '-', 0.25)
            ax.plot(angles, vals, color=colour, linewidth=lw, linestyle=ls)
            ax.fill(angles, vals, color=colour, alpha=alpha)
            plotted.append(mpatches.Patch(color=colour, label=name))
        elif row['model'] in highlight:
            (colour, label) = highlight[row['model']]
            ax.plot(angles, vals, color=colour, linewidth=1.5, linestyle='--', alpha=0.8)
            plotted.append(mpatches.Patch(color=colour, label=label, alpha=0.8))
    ax.legend(handles=plotted, loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=8)
    ax.set_title('Model Comparison — Key Metrics', size=12, pad=20)
    out = os.path.join(RESULTS_DIR, 'final_comparison_radar.png')
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'Radar-> {out}')

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--finetune_config', default='full', help='Which fine-tuned config to include (default: full)')
    return p.parse_args()

def main():
    args = parse_args()
    orig = load_original_results()
    ft = load_finetune_results(config_name=args.finetune_config)
    if not any(orig.values()):
        print('[warn] No original results found in results/. Run compare_results.py first.')
    if not ft:
        print(f"[warn] No fine-tuned results found for config='{args.finetune_config}'. Run train_finetune.py first.")
    rows = build_rows(orig, ft)
    text = print_table(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_txt(text)
    save_csv(rows)
    save_bar_chart(rows)
    save_radar_chart(rows)
    print(f'\nAll outputs saved to {RESULTS_DIR}/')
if __name__ == '__main__':
    main()
