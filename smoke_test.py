import config as cfg
cfg.PHASE1_EPOCHS = 1
cfg.PHASE2_EPOCHS = 1
cfg.PHASE3_EPOCHS = 1
cfg.BATCH_SIZE = 16
from data_loader import build_datasets
from train import train_one_run
import os
(train_ds, val_ds, test_ds, n_train, n_val, class_weights) = build_datasets()
models_to_test = [('squeezenet', 'SqueezeNet (from-scratch)'), ('mobilenetv2', 'MobileNetV2 (pretrained)')]
for (model_name, label) in models_to_test:
    ckpt = os.path.join(cfg.CHECKPOINTS_DIR, f'{model_name}_run1_best.h5')
    if os.path.exists(ckpt):
        print(f'\n[skip] {label} checkpoint already exists at {ckpt}')
        continue
    print(f"\n{'=' * 60}")
    print(f'  SMOKE TEST: {label}')
    print(f"{'=' * 60}")
    result = train_one_run(model_name, 1, train_ds, val_ds, test_ds, n_train, class_weights)
    print(f"\n[PASS] {label}  best_val_acc={result['best_val_acc']:.4f}")
print('\n\nAll smoke tests passed. Pipeline is ready for full training.')
print('Run:  python run_all.py')
