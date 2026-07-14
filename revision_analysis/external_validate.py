import os, sys, json, glob
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
PROJ = 'd:/PhD/Experiments/classify_polyp'
sys.path.insert(0, PROJ)
os.chdir(PROJ)
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
from finetune.evaluate_finetune import _load_ft_model, _tta_predict
from finetune.models import FT_REGISTRY
from finetune.train_finetune import _load_weights_by_name
from finetune.config_finetune import CHECKPOINTS_DIR
SP = 'C:/Users/yrdde/AppData/Local/Temp/claude/d--PhD-Experiments-classify-polyp/d5982003-bfbd-41f4-9d83-acfeabbb7c31/scratchpad'
ROOT = 'data/external_data/hyper-kvasir-labeled-images'
CLS = ['Gerd', 'Gerd_Normal', 'Polyp', 'Polyp_Normal']
MAP = {0: ['upper-gi-tract/pathological-findings/esophagitis-a', 'upper-gi-tract/pathological-findings/esophagitis-b-d', 'upper-gi-tract/pathological-findings/barretts', 'upper-gi-tract/pathological-findings/barretts-short-segment'], 1: ['upper-gi-tract/anatomical-landmarks/z-line', 'upper-gi-tract/anatomical-landmarks/pylorus'], 2: ['lower-gi-tract/pathological-findings/polyps'], 3: ['lower-gi-tract/anatomical-landmarks/cecum', 'lower-gi-tract/anatomical-landmarks/ileum']}

def collect():
    (paths, labels) = ([], [])
    for (lab, dirs) in MAP.items():
        for d in dirs:
            for ext in ('*.jpg', '*.jpeg', '*.png'):
                for p in glob.glob(os.path.join(ROOT, d, ext)):
                    paths.append(p)
                    labels.append(lab)
    return (paths, np.array(labels))

def build_ds(paths, size):

    def parse(p):
        raw = tf.io.read_file(p)
        img = tf.image.decode_image(raw, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, [size, size])
        return tf.cast(img, tf.float32) / 255.0
    return tf.data.Dataset.from_tensor_slices(paths).map(parse).batch(32)

def load_model(mk, run=1):
    if mk == 'efficientnet_b1':
        (m, ck, size) = _load_ft_model(mk, 'full', run)
        return (m, size)
    (m, _) = FT_REGISTRY[mk](run_seed=42 + (run - 1), image_size=256, rich_head=True, backbone_weights='imagenet')
    m.trainable = True
    ck = os.path.join(CHECKPOINTS_DIR, f'{mk}_full_run{run}_best.h5')
    try:
        m.load_weights(ck)
    except Exception:
        _load_weights_by_name(m, ck)
    return (m, 256)

def metrics(y, yp, prob):
    return dict(accuracy=float(accuracy_score(y, yp)), auc_macro=float(roc_auc_score(np.eye(4)[y], prob, average='macro', multi_class='ovr')), f1_macro=float(f1_score(y, yp, average='macro', zero_division=0)), precision_macro=float(precision_score(y, yp, average='macro', zero_division=0)), recall_macro=float(recall_score(y, yp, average='macro', zero_division=0)), per_class_recall={CLS[i]: float(recall_score(y == i, yp == i, zero_division=0)) for i in range(4)}, confusion=confusion_matrix(y, yp, labels=[0, 1, 2, 3]).tolist())
(paths, y) = collect()
print('external images:', len(y), 'per-class:', np.bincount(y).tolist(), flush=True)
OUT = {}
OUT['class_counts'] = {CLS[i]: int((y == i).sum()) for i in range(4)}
for mk in ('efficientnet_b1', 'mobilenetv3_small'):
    (model, size) = load_model(mk, 1)
    ds = build_ds(paths, size)
    probs = []
    for imgs in ds:
        probs.append(model(imgs, training=False).numpy())
    probs = np.concatenate(probs)
    yp = probs.argmax(1)
    sp = metrics(y, yp, probs)
    ds_lab = tf.data.Dataset.zip((build_ds(paths, size), tf.data.Dataset.from_tensor_slices(tf.one_hot(y, 4)).batch(32)))
    tf.random.set_seed(1234)
    (tprob, ytrue) = _tta_predict(model, ds_lab, 10)
    tp = metrics(ytrue, tprob.argmax(1), tprob)
    OUT[mk] = dict(single_pass=sp, tta10=tp)
    print(f"{mk}: single acc={sp['accuracy']:.4f} AUC={sp['auc_macro']:.4f} | tta acc={tp['accuracy']:.4f} AUC={tp['auc_macro']:.4f}", flush=True)
    tf.keras.backend.clear_session()
json.dump(OUT, open(os.path.join(SP, 'external_results.json'), 'w'), indent=2)
print('SAVED external_results.json', flush=True)
