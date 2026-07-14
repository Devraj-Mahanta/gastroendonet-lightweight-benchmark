import os, sys, numpy as np
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
PROJ = 'd:/PhD/Experiments/classify_polyp'
sys.path.insert(0, PROJ)
os.chdir(PROJ)
import tensorflow as tf
from finetune.models import FT_REGISTRY
from finetune.config_finetune import CHECKPOINTS_DIR
from finetune.train_finetune import _load_weights_by_name
from finetune.evaluate_finetune import _tta_predict
from finetune.data_finetune import build_datasets_ft
OUT = 'C:/Users/yrdde/AppData/Local/Temp/claude/d--PhD-Experiments-classify-polyp/d5982003-bfbd-41f4-9d83-acfeabbb7c31/scratchpad/preds'
for run in (1, 2, 3):
    seed = 42 + (run - 1)
    (model, _) = FT_REGISTRY['mobilenetv3_small'](run_seed=seed, image_size=256, rich_head=True, backbone_weights='imagenet')
    model.trainable = True
    ck = os.path.join(CHECKPOINTS_DIR, f'mobilenetv3_small_full_run{run}_best.h5')
    try:
        model.load_weights(ck)
    except Exception:
        _load_weights_by_name(model, ck)
    (_, _, test_ds, _, _, _) = build_datasets_ft(image_size=256, batch_size=32)
    tf.random.set_seed(1234)
    (probs, y_true) = _tta_predict(model, test_ds, 10)
    acc = float((probs.argmax(1) == y_true).mean())
    np.savez(os.path.join(OUT, f'mobilenetv3_small_full_run{run}_tta.npz'), probs=probs, y_pred=probs.argmax(1), y_true=y_true)
    print(f'mnv3 full run{run} rich-head TTA acc={acc:.4f}', flush=True)
    tf.keras.backend.clear_session()
print('DONE', flush=True)
