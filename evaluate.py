import argparse
import json
import os
import time
import numpy as np
import tensorflow as tf
import config as cfg
from checkpoint_utils import checkpoint_looks_complete
from data_loader import build_datasets
from models import MODEL_REGISTRY, MODEL_DISPLAY_NAMES
import utils

def _build_model_for_weights(model_name: str, run_id: int):
    tf.keras.backend.clear_session()
    run_seed = cfg.SEED + (run_id - 1)
    factory = MODEL_REGISTRY[model_name]
    for backbone_w in ('imagenet', None):
        try:
            (model, _) = factory(run_seed=run_seed, backbone_weights=backbone_w)
            return model
        except Exception:
            continue
    (model, _) = factory(run_seed=run_seed)
    return model

def _load_weights_by_name(model, ckpt_path: str) -> int:
    import h5py
    weight_map: dict = {}
    with h5py.File(ckpt_path, 'r') as h5:
        for group_name in h5.keys():
            if group_name == 'top_level_model_weights':
                continue
            g = h5[group_name]
            raw_names = g.attrs.get('weight_names', [])
            for raw in raw_names:
                wname = raw.decode('utf-8') if isinstance(raw, bytes) else str(raw)
                parts = wname.split('/')
                node = g
                try:
                    for p in parts:
                        node = node[p]
                    arr = node[()] if node.shape == () else node[:]
                    weight_map[wname] = arr
                except Exception:
                    pass
    loaded = 0
    for var in model.variables:
        parts = var.name.split('/')
        for start in range(len(parts) - 1):
            key = '/'.join(parts[start:])
            if key in weight_map:
                arr = weight_map[key]
                if var.shape == arr.shape:
                    var.assign(arr)
                    loaded += 1
                else:
                    print(f'  [warn] shape mismatch for {key}: model={var.shape} vs ckpt={arr.shape}')
                break
    return loaded

def load_checkpoint_model(model_name: str, run_id: int, ckpt_path: str):
    model = _build_model_for_weights(model_name, run_id)
    total = len(model.variables)
    try:
        model.load_weights(ckpt_path)
        return model
    except Exception:
        pass
    n = _load_weights_by_name(model, ckpt_path)
    if n < total // 2:
        raise RuntimeError(f'Could not load checkpoint: only {n}/{total} variables matched ({ckpt_path})')
    missed = total - n
    if missed:
        print(f'  [{missed} variables unmatched — check checkpoint compatibility]')
    return model

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray):
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
    acc = accuracy_score(y_true, y_pred)
    f1_m = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_w = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_pc = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    try:
        y_oh = np.eye(cfg.NUM_CLASSES)[y_true]
        auc = roc_auc_score(y_oh, y_prob, average='macro', multi_class='ovr')
    except Exception:
        auc = float('nan')
    cm = confusion_matrix(y_true, y_pred, labels=list(range(cfg.NUM_CLASSES))).tolist()
    return {'accuracy': float(acc), 'auc_macro': float(auc), 'f1_macro': float(f1_m), 'f1_weighted': float(f1_w), 'f1_per_class': {cfg.CLASS_NAMES[i]: float(v) for (i, v) in enumerate(f1_pc)}, 'precision_macro': float(prec), 'recall_macro': float(rec), 'confusion_matrix': cm}

def measure_inference_time(model, test_ds, n_warmup=5):
    batch_times = []
    batch_size = cfg.BATCH_SIZE
    for (i, (imgs, _)) in enumerate(test_ds):
        t0 = time.perf_counter()
        _ = model(imgs, training=False)
        t1 = time.perf_counter()
        if i >= n_warmup:
            batch_times.append((t1 - t0) / imgs.shape[0] * 1000)
    return float(np.mean(batch_times)) if batch_times else float('nan')

def evaluate_checkpoint(model_name: str, run_id: int, test_ds, verbose=True):
    ckpt_path = os.path.join(cfg.CHECKPOINTS_DIR, f'{model_name}_run{run_id}_best.h5')
    (ok, reason) = checkpoint_looks_complete(model_name, ckpt_path)
    if not ok:
        print(f'[skip] {ckpt_path} {reason}')
        return None
    print(f'\nEvaluating: {MODEL_DISPLAY_NAMES[model_name]}  run={run_id}')
    try:
        model = load_checkpoint_model(model_name, run_id, ckpt_path)
    except RuntimeError as exc:
        print(f'[skip] {exc}')
        return None
    (all_probs, all_true) = ([], [])
    for (imgs, labels) in test_ds:
        probs = model(imgs, training=False).numpy()
        labels = labels.numpy()
        all_probs.append(probs)
        all_true.append(np.argmax(labels, axis=-1))
    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_true)
    y_pred = np.argmax(y_prob, axis=-1)
    metrics = compute_metrics(y_true, y_pred, y_prob)
    inf_ms = measure_inference_time(model, test_ds)
    model_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
    n_params = model.count_params()
    result = {'model': model_name, 'display_name': MODEL_DISPLAY_NAMES[model_name], 'run': run_id, 'checkpoint': ckpt_path, 'n_params': n_params, 'model_size_mb': round(model_mb, 2), 'inference_ms_per_image': round(inf_ms, 4), **metrics}
    if verbose:
        print(f"  accuracy      = {metrics['accuracy']:.4f}")
        print(f"  auc (macro)   = {metrics['auc_macro']:.4f}")
        print(f"  f1 (macro)    = {metrics['f1_macro']:.4f}")
        print(f"  precision     = {metrics['precision_macro']:.4f}")
        print(f"  recall        = {metrics['recall_macro']:.4f}")
        print(f'  inference     = {inf_ms:.2f} ms/image')
        print(f'  model size    = {model_mb:.1f} MB   params = {n_params:,}')
    out_path = os.path.join(cfg.RESULTS_DIR, f'{model_name}_run{run_id}_eval.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    utils.plot_confusion_matrix(model, test_ds, model_name, run_id)
    return result

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='all', help=f"Model name or 'all'. Choices: {list(MODEL_REGISTRY.keys())}")
    p.add_argument('--run', type=int, default=0, help='Run index (1-3) or 0 for all runs')
    return p.parse_args()

def main():
    args = parse_args()
    (_, _, test_ds, *_) = build_datasets()
    models = list(MODEL_REGISTRY.keys()) if args.model == 'all' else [args.model]
    runs = [1, 2, 3] if args.run == 0 else [args.run]
    for m in models:
        for r in runs:
            evaluate_checkpoint(m, r, test_ds)
    print('\nDone. Run  python compare_results.py  to see the summary table.')
if __name__ == '__main__':
    main()
