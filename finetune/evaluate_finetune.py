import sys, os
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)
import argparse, json, time
import numpy as np
import tensorflow as tf
import config as base_cfg
from finetune.config_finetune import ABLATION_CONFIGS, CHECKPOINTS_DIR, RESULTS_DIR, BASE_IMAGE_SIZE, HIGH_IMAGE_SIZE, BATCH_SIZE, TTA_N_AUGMENTS, NUM_RUNS
from finetune.models import FT_REGISTRY, FT_DISPLAY_NAMES

def _make_tta_aug():
    return tf.keras.Sequential([tf.keras.layers.RandomFlip('horizontal_and_vertical'), tf.keras.layers.RandomRotation(0.1), tf.keras.layers.RandomZoom((-0.1, 0.1))])

def _tta_predict(model, test_ds, n_aug=TTA_N_AUGMENTS):
    cumulative = None
    y_true = None
    tta_aug = _make_tta_aug()
    for i in range(n_aug):
        (run_probs, run_true) = ([], [])
        for (imgs, labels) in test_ds:
            if i > 0:
                imgs = tta_aug(imgs, training=True)
            probs = model(imgs, training=False).numpy()
            run_probs.append(probs)
            if y_true is None:
                run_true.append(np.argmax(labels.numpy(), axis=-1))
        batch_probs = np.concatenate(run_probs)
        if cumulative is None:
            cumulative = batch_probs
            y_true = np.concatenate(run_true) if run_true else y_true
        else:
            cumulative += batch_probs
    return (cumulative / n_aug, y_true)

def _compute_metrics(y_true, y_pred, y_prob):
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1w = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    f1pc = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    try:
        y_oh = np.eye(base_cfg.NUM_CLASSES)[y_true]
        auc = roc_auc_score(y_oh, y_prob, average='macro', multi_class='ovr')
    except Exception:
        auc = float('nan')
    cm = confusion_matrix(y_true, y_pred, labels=list(range(base_cfg.NUM_CLASSES))).tolist()
    return dict(accuracy=float(acc), auc_macro=float(auc), f1_macro=float(f1m), f1_weighted=float(f1w), f1_per_class={base_cfg.CLASS_NAMES[i]: float(v) for (i, v) in enumerate(f1pc)}, precision_macro=float(prec), recall_macro=float(rec), confusion_matrix=cm)

def _bootstrap_ci(y_true, y_pred, y_prob, n_boot=1000, ci=0.95):
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
    rng = np.random.default_rng(42)
    N = len(y_true)
    (accs, aucs, f1s) = ([], [], [])
    for _ in range(n_boot):
        idx = rng.integers(0, N, N)
        yt = y_true[idx]
        yp = y_pred[idx]
        ypr = y_prob[idx]
        accs.append(accuracy_score(yt, yp))
        f1s.append(f1_score(yt, yp, average='macro', zero_division=0))
        try:
            yoh = np.eye(base_cfg.NUM_CLASSES)[yt]
            aucs.append(roc_auc_score(yoh, ypr, average='macro', multi_class='ovr'))
        except Exception:
            aucs.append(float('nan'))
    (lo, hi) = ((1 - ci) / 2, 1 - (1 - ci) / 2)

    def _ci(arr):
        arr = np.array(arr)
        arr = arr[~np.isnan(arr)]
        return (float(np.quantile(arr, lo)), float(np.quantile(arr, hi)))
    return {'accuracy_ci95': _ci(accs), 'auc_ci95': _ci(aucs), 'f1_macro_ci95': _ci(f1s)}

def evaluate_model(model, test_ds, model_name, cfg_name, run_id, ckpt_path=None, use_tta=True):
    t0 = time.perf_counter()
    if use_tta:
        (y_prob, y_true) = _tta_predict(model, test_ds, n_aug=TTA_N_AUGMENTS)
    else:
        (all_p, all_t) = ([], [])
        for (imgs, labels) in test_ds:
            all_p.append(model(imgs, training=False).numpy())
            all_t.append(np.argmax(labels.numpy(), axis=-1))
        y_prob = np.concatenate(all_p)
        y_true = np.concatenate(all_t)
    t_eval = time.perf_counter() - t0
    y_pred = np.argmax(y_prob, axis=-1)
    metrics = _compute_metrics(y_true, y_pred, y_prob)
    ci = _bootstrap_ci(y_true, y_pred, y_prob)
    inf_ms = t_eval / len(y_true) * 1000 / (TTA_N_AUGMENTS if use_tta else 1)
    model_mb = os.path.getsize(ckpt_path) / (1024 * 1024) if ckpt_path and os.path.exists(ckpt_path) else None
    result = dict(model=model_name, config=cfg_name, run=run_id, checkpoint=ckpt_path, n_params=model.count_params(), model_size_mb=round(model_mb, 2) if model_mb else None, inference_ms_per_image=round(inf_ms, 4), tta_augments=TTA_N_AUGMENTS if use_tta else 0, **metrics, **ci)
    return result

def _load_ft_model(model_name, cfg_name, run_id):
    ft_cfg = ABLATION_CONFIGS[cfg_name]
    img_size = HIGH_IMAGE_SIZE[model_name] if ft_cfg['image_size_key'] == 'high' else BASE_IMAGE_SIZE[model_name]
    tf.keras.backend.clear_session()
    run_seed = base_cfg.SEED + (run_id - 1)
    (model, _) = FT_REGISTRY[model_name](run_seed=run_seed, image_size=img_size, rich_head=ft_cfg['rich_head'], backbone_weights='imagenet')
    model.trainable = True
    ckpt_path = os.path.join(CHECKPOINTS_DIR, f'{model_name}_{cfg_name}_run{run_id}_best.h5')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}\nRun train_finetune.py first.')
    try:
        model.load_weights(ckpt_path)
    except Exception:
        from finetune.train_finetune import _load_weights_by_name
        _load_weights_by_name(model, ckpt_path)
    return (model, ckpt_path, img_size)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='efficientnet_b1', choices=list(FT_REGISTRY.keys()) + ['all'])
    p.add_argument('--config', default='full', choices=list(ABLATION_CONFIGS.keys()) + ['all'])
    p.add_argument('--run', type=int, default=0)
    p.add_argument('--no_tta', action='store_true', help='Disable test-time augmentation')
    return p.parse_args()

def main():
    args = parse_args()
    models = list(FT_REGISTRY.keys()) if args.model == 'all' else [args.model]
    cfgs = list(ABLATION_CONFIGS.keys()) if args.config == 'all' else [args.config]
    runs = list(range(1, NUM_RUNS + 1)) if args.run == 0 else [args.run]
    from finetune.data_finetune import build_datasets_ft
    for m in models:
        for c in cfgs:
            ft_cfg = ABLATION_CONFIGS[c]
            img_size = HIGH_IMAGE_SIZE[m] if ft_cfg['image_size_key'] == 'high' else BASE_IMAGE_SIZE[m]
            (_, _, test_ds, *_) = build_datasets_ft(image_size=img_size, batch_size=BATCH_SIZE)
            for r in runs:
                out = os.path.join(RESULTS_DIR, f'{m}_{c}_run{r}_eval.json')
                if os.path.exists(out):
                    print(f'[skip] {out}')
                    continue
                print(f'\nEvaluating {FT_DISPLAY_NAMES[m]}  config={c}  run={r}')
                (model, ckpt, _) = _load_ft_model(m, c, r)
                result = evaluate_model(model, test_ds, m, c, r, ckpt_path=ckpt, use_tta=not args.no_tta)
                os.makedirs(RESULTS_DIR, exist_ok=True)
                with open(out, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"  acc={result['accuracy']:.4f}  auc={result['auc_macro']:.4f}  f1={result['f1_macro']:.4f}")
                print(f"  95% CI  acc={result['accuracy_ci95']}  auc={result['auc_ci95']}")
                print(f'  Saved -> {out}')
if __name__ == '__main__':
    main()
