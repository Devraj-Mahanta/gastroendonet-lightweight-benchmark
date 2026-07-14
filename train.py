import argparse
import json
import math
import os
import time
import numpy as np

class _NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
import tensorflow as tf
import config as cfg
from data_loader import build_datasets
from models import MODEL_REGISTRY, MODEL_DISPLAY_NAMES
import utils
tf.random.set_seed(cfg.SEED)
np.random.seed(cfg.SEED)

def _set_phase(model, base, phase: int):
    if base is None:
        model.trainable = True
        return
    if phase == 1:
        base.trainable = False
    elif phase == 2:
        base.trainable = True
        n = len(base.layers)
        cutoff = int(n * (1.0 - cfg.UNFREEZE_FRACTION))
        for (i, layer) in enumerate(base.layers):
            layer.trainable = i >= cutoff
    elif phase == 3:
        model.trainable = True

def _count_trainable(model):
    return sum((tf.size(v).numpy() for v in model.trainable_variables))

def _fit_phase(model, base, phase: int, n_epochs: int, lr: float, train_ds, val_ds, steps_per_epoch, class_weights):
    _set_phase(model, base, phase)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=cfg.LABEL_SMOOTHING), metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy'), tf.keras.metrics.AUC(name='auc')])
    trainable = _count_trainable(model)
    total = model.count_params()
    print(f"\n{'-' * 60}")
    print(f'  Phase {phase}  |  epochs={n_epochs}  LR={lr}  trainable={trainable:,} / {total:,}')
    print(f"{'-' * 60}")
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=cfg.EARLY_STOP_PATIENCE, restore_best_weights=True, verbose=1), tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-08, verbose=1)]
    hist = model.fit(train_ds, validation_data=val_ds, epochs=n_epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks, class_weight=class_weights, verbose=1)
    return hist.history

def train_one_run(model_name: str, run_id: int, train_ds, val_ds, test_ds, n_train: int, class_weights: dict):
    os.makedirs(cfg.CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    run_seed = cfg.SEED + (run_id - 1)
    (model, base) = MODEL_REGISTRY[model_name](run_seed=run_seed)
    display = MODEL_DISPLAY_NAMES[model_name]
    print(f"\n{'=' * 60}")
    print(f'  {display}  --  Run {run_id}/3')
    print(f"{'=' * 60}")
    steps_per_epoch = math.ceil(n_train / cfg.BATCH_SIZE)
    phase_configs = [(1, cfg.PHASE1_EPOCHS, cfg.PHASE1_LR), (2, cfg.PHASE2_EPOCHS, cfg.PHASE2_LR), (3, cfg.PHASE3_EPOCHS, cfg.PHASE3_LR)]
    best_val_acc = 0.0
    best_weights = model.get_weights()
    all_histories = []
    phase_labels = []
    for (phase, n_ep, lr) in phase_configs:
        history = _fit_phase(model, base, phase, n_ep, lr, train_ds, val_ds, steps_per_epoch, class_weights)
        all_histories.append(history)
        phase_labels.extend([f'ph{phase}'] * len(history['val_accuracy']))
        phase_best = float(max(history['val_accuracy']))
        print(f'Phase {phase} best val_accuracy = {phase_best:.4f}')
        if phase_best > best_val_acc:
            best_val_acc = phase_best
            best_weights = model.get_weights()
    model.set_weights(best_weights)
    model.compile(optimizer=tf.keras.optimizers.Adam(cfg.PHASE3_LR), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy'), tf.keras.metrics.AUC(name='auc')])
    print('\nEvaluating on held-out test set...')
    test_results = model.evaluate(test_ds, verbose=1)
    test_metrics = {k: float(v) for (k, v) in zip(model.metrics_names, test_results)}
    print(f"Test  loss={test_metrics['loss']:.4f}  acc={test_metrics['accuracy']:.4f}  auc={test_metrics['auc']:.4f}")
    ckpt_name = f'{model_name}_run{run_id}_best.h5'
    ckpt_path = os.path.join(cfg.CHECKPOINTS_DIR, ckpt_name)
    model.save_weights(ckpt_path)
    print(f'Saved  ->  {ckpt_path}')
    run_record = {'model': model_name, 'run': run_id, 'run_seed': run_seed, 'best_val_acc': best_val_acc, 'test_metrics': test_metrics, 'phases': all_histories}
    hist_path = os.path.join(cfg.RESULTS_DIR, f'{model_name}_run{run_id}_history.json')
    with open(hist_path, 'w') as f:
        json.dump(run_record, f, indent=2, cls=_NumpyEncoder)
    utils.plot_run_history(all_histories, model_name, run_id)
    utils.plot_confusion_matrix(model, test_ds, model_name, run_id)
    return run_record

def parse_args():
    p = argparse.ArgumentParser(description='Train one model for one run (use run_all.py for all 8 × 3)')
    p.add_argument('--model', required=True, choices=list(MODEL_REGISTRY.keys()))
    p.add_argument('--run', type=int, required=True, choices=[1, 2, 3], help='Run index (1, 2, or 3); each uses a different weight-init seed')
    return p.parse_args()

def main():
    args = parse_args()
    (train_ds, val_ds, test_ds, n_train, n_val, class_weights) = build_datasets()
    train_one_run(args.model, args.run, train_ds, val_ds, test_ds, n_train, class_weights)
if __name__ == '__main__':
    main()
