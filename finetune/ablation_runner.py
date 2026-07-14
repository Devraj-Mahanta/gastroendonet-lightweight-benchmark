import sys, os
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)
import argparse
from finetune.config_finetune import ABLATION_CONFIGS, NUM_RUNS
from finetune.models import FT_REGISTRY
from finetune.train_finetune import finetune_one_run

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='efficientnet_b1', choices=list(FT_REGISTRY.keys()) + ['all'])
    p.add_argument('--config', default='all', choices=list(ABLATION_CONFIGS.keys()) + ['all'])
    p.add_argument('--runs', nargs='+', type=int, default=list(range(1, NUM_RUNS + 1)))
    return p.parse_args()

def main():
    args = parse_args()
    models = list(FT_REGISTRY.keys()) if args.model == 'all' else [args.model]
    cfgs = list(ABLATION_CONFIGS.keys()) if args.config == 'all' else [args.config]
    total = len(models) * len(cfgs) * len(args.runs)
    done = 0
    print('=' * 65)
    print(f'  Ablation study — {total} training runs')
    print(f'  Models  : {models}')
    print(f'  Configs : {cfgs}')
    print(f'  Runs    : {args.runs}')
    print('=' * 65)
    results_summary = {}
    failed_runs = []
    for model_name in models:
        results_summary[model_name] = {}
        for cfg_name in cfgs:
            ft_cfg = ABLATION_CONFIGS[cfg_name]
            run_results = []
            for run_id in args.runs:
                done += 1
                print(f'\n[{done}/{total}]')
                try:
                    r = finetune_one_run(model_name, run_id, cfg_name, ft_cfg)
                except Exception as exc:
                    import traceback
                    label = f'{model_name}/{cfg_name}/run{run_id}'
                    print(f'\n[ERROR] {label} failed — skipping:')
                    traceback.print_exc()
                    failed_runs.append(label)
                    r = None
                if r:
                    run_results.append(r)
            if run_results:
                import numpy as np
                accs = [r['accuracy'] for r in run_results]
                aucs = [r['auc_macro'] for r in run_results]
                f1s = [r['f1_macro'] for r in run_results]
                results_summary[model_name][cfg_name] = {'acc_mean': float(np.mean(accs)), 'acc_std': float(np.std(accs)), 'auc_mean': float(np.mean(aucs)), 'auc_std': float(np.std(aucs)), 'f1_mean': float(np.mean(f1s)), 'f1_std': float(np.std(f1s))}
    print('\n' + '=' * 75)
    print('  ABLATION SUMMARY')
    print('=' * 75)
    hdr = f"{'Config':<18}  {'Description':<42}  {'Acc':>7}  {'AUC':>7}  {'F1':>7}"
    print(hdr)
    print('-' * 75)
    for model_name in models:
        print(f'\n  Model: {model_name}')
        for (cfg_name, vals) in results_summary.get(model_name, {}).items():
            desc = ABLATION_CONFIGS[cfg_name]['description'][:42]
            row = f"  {cfg_name:<16}  {desc:<42}  {vals['acc_mean']:.4f}  {vals['auc_mean']:.4f}  {vals['f1_mean']:.4f}"
            print(row)
    print('=' * 75)
    if failed_runs:
        print(f'\n  WARNING: {len(failed_runs)} run(s) failed and were skipped:')
        for label in failed_runs:
            print(f'    {label}')
        print('  Re-run the script to retry failed runs.')
    print('\nNext step:')
    print('  python finetune/compare_finetune.py --model', args.model)
if __name__ == '__main__':
    main()
