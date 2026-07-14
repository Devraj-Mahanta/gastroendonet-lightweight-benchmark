import argparse
import os
import subprocess
import sys
import config as cfg
from checkpoint_utils import checkpoint_looks_complete
from models import MODEL_REGISTRY

def checkpoint_exists(model_name: str, run_id: int) -> bool:
    path = os.path.join(cfg.CHECKPOINTS_DIR, f'{model_name}_run{run_id}_best.h5')
    (ok, reason) = checkpoint_looks_complete(model_name, path)
    if not ok:
        if reason != 'not found':
            print(f'[warn] Ignoring checkpoint: {path} ({reason})')
        return False
    return True

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--models', nargs='+', default=list(MODEL_REGISTRY.keys()), choices=list(MODEL_REGISTRY.keys()))
    p.add_argument('--runs', nargs='+', type=int, default=[1, 2, 3], choices=[1, 2, 3])
    p.add_argument('--force', action='store_true', help='Re-train even if checkpoint already exists')
    return p.parse_args()

def main():
    args = parse_args()
    if not os.path.exists(cfg.SPLIT_FILE):
        print('Fixed split not found -- generating now...')
        subprocess.run([sys.executable, 'generate_split.py'], check=True)
    total = len(args.models) * len(args.runs)
    done = 0
    for model_name in args.models:
        for run_id in args.runs:
            done += 1
            tag = f'[{done}/{total}]'
            if not args.force and checkpoint_exists(model_name, run_id):
                print(f'{tag} SKIP  {model_name} run={run_id}  (checkpoint exists)')
                continue
            print(f'\n{tag} START  {model_name}  run={run_id}')
            result = subprocess.run([sys.executable, 'train.py', '--model', model_name, '--run', str(run_id)], check=False)
            if result.returncode != 0:
                print(f'{tag} ERROR  {model_name} run={run_id} exited with code {result.returncode}')
                print('      Continuing with next model...')
    print('\n' + '=' * 60)
    print('  All runs complete -- running evaluation...')
    print('=' * 60)
    subprocess.run([sys.executable, 'evaluate.py', '--model', 'all', '--run', '0'])
    print('\n' + '=' * 60)
    print('  Building comparison table...')
    print('=' * 60)
    subprocess.run([sys.executable, 'compare_results.py'])
    print('\nAll done.  Results in:', cfg.RESULTS_DIR)
if __name__ == '__main__':
    main()
