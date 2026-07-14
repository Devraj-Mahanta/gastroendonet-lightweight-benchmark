import sys, os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('TF_XLA_FLAGS', '--tf_xla_auto_jit=0')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)
import argparse, json, math
import numpy as np
import tensorflow as tf
import config as base_cfg
from finetune.config_finetune import ABLATION_CONFIGS, CHECKPOINTS_DIR, RESULTS_DIR, BASE_IMAGE_SIZE, HIGH_IMAGE_SIZE, BATCH_SIZE, WARMUP_EPOCHS, FINETUNE_EPOCHS, EARLY_STOP_PATIENCE, INITIAL_LR, MIN_LR, WEIGHT_DECAY, HEAD_WARMUP_EPOCHS, FOCAL_GAMMA, LABEL_SMOOTHING, NUM_RUNS
from finetune.data_finetune import build_datasets_ft
from finetune.losses import CategoricalFocalLoss
from finetune.schedulers import WarmupCosineDecay
from finetune.models import FT_REGISTRY, FT_DISPLAY_NAMES, PRETRAINED_CKPTS

def _load_weights_by_name(model, ckpt_path):
    import h5py
    weight_map = {}
    with h5py.File(ckpt_path, 'r') as h5:
        for gname in h5.keys():
            if gname == 'top_level_model_weights':
                continue
            g = h5[gname]
            for raw in g.attrs.get('weight_names', []):
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
                break
    return loaded
_BY_NAME_MODELS = {'mobilenetv3_small', 'mobilenetv3'}

def load_pretrained_checkpoint(model, model_name, run_id):
    ckpt_template = PRETRAINED_CKPTS[model_name]
    ckpt_path = ckpt_template.format(run=run_id)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Original checkpoint not found: {ckpt_path}\nRun the original training first (run_all.py).')
    if model_name in _BY_NAME_MODELS:
        n = _load_weights_by_name(model, ckpt_path)
        print(f'  [ckpt] Loaded {n}/{len(model.variables)} vars by name from {ckpt_path}')
    else:
        try:
            model.load_weights(ckpt_path)
            print(f'  [ckpt] Loaded {ckpt_path} (positional)')
        except Exception:
            n = _load_weights_by_name(model, ckpt_path)
            print(f'  [ckpt] Loaded {n}/{len(model.variables)} vars by name from {ckpt_path}')

def build_loss(ft_cfg):
    if ft_cfg['focal_loss']:
        return CategoricalFocalLoss(gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTHING)
    return tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING)

def _compile(model, lr, ft_cfg):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=build_loss(ft_cfg), metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy'), tf.keras.metrics.AUC(name='auc')])

def _callbacks(ckpt_out, ft_cfg):
    cbs = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=EARLY_STOP_PATIENCE, restore_best_weights=True, verbose=1), tf.keras.callbacks.ModelCheckpoint(ckpt_out, monitor='val_accuracy', save_best_only=True, save_weights_only=True, verbose=0)]
    if ft_cfg['schedule'] != 'cosine':
        cbs.append(tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-08, verbose=1))
    return cbs

def _lr_for_phase(ft_cfg, steps_per_epoch, total_epochs, base_lr=INITIAL_LR):
    if ft_cfg['schedule'] == 'cosine':
        warmup_steps = steps_per_epoch * WARMUP_EPOCHS
        total_steps = steps_per_epoch * total_epochs
        return WarmupCosineDecay(base_lr, warmup_steps, total_steps, MIN_LR)
    return base_lr

def finetune_one_run(model_name, run_id, cfg_name, ft_cfg):
    display = f'{FT_DISPLAY_NAMES[model_name]}  cfg={cfg_name}  run={run_id}'
    print(f"\n{'=' * 65}")
    print(f'  {display}')
    print(f"  {ft_cfg['description']}")
    print(f"{'=' * 65}")
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_path = os.path.join(RESULTS_DIR, f'{model_name}_{cfg_name}_run{run_id}_eval.json')
    if os.path.exists(result_path):
        print(f'  [skip] Already done: {result_path}')
        with open(result_path) as f:
            return json.load(f)
    ckpt_out = os.path.join(CHECKPOINTS_DIR, f'{model_name}_{cfg_name}_run{run_id}_best.h5')
    img_size = HIGH_IMAGE_SIZE[model_name] if ft_cfg['image_size_key'] == 'high' else BASE_IMAGE_SIZE[model_name]
    (train_ds, val_ds, test_ds, n_train, _, class_wts) = build_datasets_ft(image_size=img_size, batch_size=BATCH_SIZE, mixup_alpha=ft_cfg['mixup_alpha'], cutmix_alpha=ft_cfg['cutmix_alpha'], strong_aug=ft_cfg['strong_aug'])
    steps_per_epoch = math.ceil(n_train / BATCH_SIZE)
    tf.keras.backend.clear_session()
    run_seed = base_cfg.SEED + (run_id - 1)
    factory = FT_REGISTRY[model_name]
    wd = WEIGHT_DECAY if ft_cfg['adamw'] else 0.0
    (model, base) = factory(run_seed=run_seed, image_size=img_size, rich_head=ft_cfg['rich_head'], backbone_weights='imagenet', weight_decay=wd)
    if ft_cfg['init_from'] == 'checkpoint':
        load_pretrained_checkpoint(model, model_name, run_id)
    if ft_cfg['rich_head']:
        base.trainable = False
        _compile(model, lr=0.0003, ft_cfg=ft_cfg)
        print(f'\n  Phase A — head warm-up  ({HEAD_WARMUP_EPOCHS} epochs, backbone frozen)')
        model.fit(train_ds, validation_data=val_ds, epochs=HEAD_WARMUP_EPOCHS, class_weight=class_wts, verbose=1, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1)])
    model.trainable = True
    lr = _lr_for_phase(ft_cfg, steps_per_epoch, FINETUNE_EPOCHS)
    _compile(model, lr=lr, ft_cfg=ft_cfg)
    print(f"\n  Phase B — full model  (schedule={ft_cfg['schedule']}, max {FINETUNE_EPOCHS} epochs)")
    model.fit(train_ds, validation_data=val_ds, epochs=FINETUNE_EPOCHS, callbacks=_callbacks(ckpt_out, ft_cfg), class_weight=class_wts, verbose=1)
    if os.path.exists(ckpt_out):
        if model_name in _BY_NAME_MODELS:
            _load_weights_by_name(model, ckpt_out)
        else:
            try:
                model.load_weights(ckpt_out)
            except Exception:
                _load_weights_by_name(model, ckpt_out)
    from finetune.evaluate_finetune import evaluate_model
    result = evaluate_model(model, test_ds, model_name, cfg_name, run_id, ckpt_path=ckpt_out)
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n  accuracy={result['accuracy']:.4f}  auc={result['auc_macro']:.4f}  f1={result['f1_macro']:.4f}")
    print(f'  Saved -> {result_path}')
    return result

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='efficientnet_b1', choices=list(FT_REGISTRY.keys()) + ['all'])
    p.add_argument('--config', default='full', choices=list(ABLATION_CONFIGS.keys()) + ['all'])
    p.add_argument('--run', type=int, default=0, help='Run 1-3, or 0 for all runs')
    return p.parse_args()

def main():
    args = parse_args()
    models = list(FT_REGISTRY.keys()) if args.model == 'all' else [args.model]
    cfgs = list(ABLATION_CONFIGS.keys()) if args.config == 'all' else [args.config]
    runs = list(range(1, NUM_RUNS + 1)) if args.run == 0 else [args.run]
    total = len(models) * len(cfgs) * len(runs)
    done = 0
    for m in models:
        for c in cfgs:
            for r in runs:
                done += 1
                print(f'\n[{done}/{total}]', end=' ')
                finetune_one_run(m, r, c, ABLATION_CONFIGS[c])
    print('\nAll fine-tuning complete.')
    print('Next: python finetune/compare_finetune.py --model', args.model)
if __name__ == '__main__':
    main()
