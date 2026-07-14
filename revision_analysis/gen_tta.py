import os, sys, numpy as np
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
PROJ = 'd:/PhD/Experiments/classify_polyp'
sys.path.insert(0, PROJ)
os.chdir(PROJ)
import tensorflow as tf
from finetune.evaluate_finetune import _load_ft_model, _tta_predict
from finetune.data_finetune import build_datasets_ft
OUT = 'C:/Users/yrdde/AppData/Local/Temp/claude/d--PhD-Experiments-classify-polyp/d5982003-bfbd-41f4-9d83-acfeabbb7c31/scratchpad/preds'
os.makedirs(OUT, exist_ok=True)
JOBS = [('efficientnet_b1', 'full', 1), ('efficientnet_b1', 'full', 2), ('efficientnet_b1', 'full', 3), ('mobilenetv3_small', 'full', 1), ('mobilenetv3_small', 'full', 2), ('mobilenetv3_small', 'full', 3), ('efficientnet_b1', 'baseline', 1), ('efficientnet_b1', 'a2_cosine', 1), ('efficientnet_b1', 'a3_mixup', 1)]
for (mk, cf, rn) in JOBS:
    tag = f'{mk}_{cf}_run{rn}_tta'
    fp = os.path.join(OUT, tag + '.npz')
    if os.path.exists(fp):
        print('skip', tag, flush=True)
        continue
    (model, ckpt, size) = _load_ft_model(mk, cf, rn)
    (_, _, test_ds, _, _, _) = build_datasets_ft(image_size=size, batch_size=32)
    tf.random.set_seed(1234)
    (probs, y_true) = _tta_predict(model, test_ds, 10)
    y_pred = probs.argmax(1)
    acc = float((y_pred == y_true).mean())
    np.savez(fp, probs=probs, y_pred=y_pred, y_true=y_true)
    print(f'{tag}: size={size} acc={acc:.4f}', flush=True)
    tf.keras.backend.clear_session()
print('DONE', flush=True)
