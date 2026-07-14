import os, sys, json, glob
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
PROJ = 'd:/PhD/Experiments/classify_polyp'
sys.path.insert(0, PROJ)
os.chdir(PROJ)
import numpy as np
import tensorflow as tf
from PIL import Image
from finetune.evaluate_finetune import _load_ft_model
SP = 'C:/Users/yrdde/AppData/Local/Temp/claude/d--PhD-Experiments-classify-polyp/d5982003-bfbd-41f4-9d83-acfeabbb7c31/scratchpad'
IMG = 'data/external_data/hyper-kvasir-segmentation/images'
MSK = 'data/external_data/hyper-kvasir-segmentation/masks'
SIZE = 240
POLYP = 2
(model, ck, size) = _load_ft_model('efficientnet_b1', 'full', 1)
rescale = model.get_layer('rescale_to_0_255')
gap = model.get_layer('gap')
classifier = model.get_layer('classifier')
base = next((l for l in model.layers if 'efficientnet' in l.name.lower()))

@tf.function
def batch_cam(imgs):
    with tf.GradientTape() as tape:
        feat = base(rescale(imgs), training=False)
        tape.watch(feat)
        logits = classifier(gap(feat))
        score = tf.reduce_sum(logits[:, POLYP])
    grads = tape.gradient(score, feat)
    pooled = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.reduce_sum(feat * pooled[:, None, None, :], -1)
    cam = tf.maximum(cam, 0.0)
    cam = cam / (tf.reduce_max(cam, axis=(1, 2), keepdims=True) + 1e-08)
    cam = tf.image.resize(cam[..., None], [SIZE, SIZE])[..., 0]
    return (cam, logits)

@tf.function
def batch_prob(imgs):
    return model(imgs, training=False)[:, POLYP]

def load_img(p):
    return np.asarray(Image.open(p).convert('RGB').resize((SIZE, SIZE), Image.BILINEAR), np.float32) / 255.0

def load_mask(p):
    return (np.asarray(Image.open(p).convert('L').resize((SIZE, SIZE), Image.NEAREST)) > 127).astype(np.float32)
files = sorted(glob.glob(os.path.join(IMG, '*.jpg')))
print('polyp seg images:', len(files), flush=True)
pg = eng = iou = 0.0
ncnt = 0
B = 32
(buf_img, buf_mask) = ([], [])

def flush(buf_img, buf_mask):
    global pg, eng, iou, ncnt
    (cams, _) = batch_cam(tf.stack(buf_img))
    cams = cams.numpy()
    for (hm, mask) in zip(cams, buf_mask):
        yx = np.unravel_index(np.argmax(hm), hm.shape)
        pg += float(mask[yx] > 0.5)
        eng += float((hm * mask).sum() / (hm.sum() + 1e-08))
        thr = hm >= 0.5 * hm.max()
        inter = float((thr & (mask > 0.5)).sum())
        union = float((thr | (mask > 0.5)).sum())
        iou += inter / (union + 1e-08)
        ncnt += 1
for (i, f) in enumerate(files):
    mp = os.path.join(MSK, os.path.basename(f))
    if not os.path.exists(mp):
        continue
    m = load_mask(mp)
    if m.sum() < 1:
        continue
    buf_img.append(load_img(f))
    buf_mask.append(m)
    if len(buf_img) == B:
        flush(buf_img, buf_mask)
        (buf_img, buf_mask) = ([], [])
        if ncnt % 160 == 0:
            print('  pointing-game processed', ncnt, flush=True)
if buf_img:
    flush(buf_img, buf_mask)
res = dict(n=ncnt, pointing_game=round(pg / ncnt, 4), energy_pointing_game=round(eng / ncnt, 4), cam_mask_iou=round(iou / ncnt, 4))
print('POINTING:', res, flush=True)
STEPS = 20
NSAMP = 150
rng = np.random.default_rng(0)
samp = rng.choice(len(files), size=min(NSAMP, len(files)), replace=False)
(dels, inss) = ([], [])
dacc = np.zeros(STEPS + 1)
iacc = np.zeros(STEPS + 1)
cc = 0
for (j, idx) in enumerate(samp):
    img = load_img(files[idx])
    (cam, _) = batch_cam(tf.stack([img]))
    hm = cam.numpy()[0]
    order = np.argsort(hm.ravel())[::-1]
    flat = img.reshape(-1, 3)
    blur = np.full_like(flat, flat.mean(0))
    npix = len(order)
    chunk = max(1, npix // STEPS)
    (dbatch, ibatch) = ([], [])
    for s in range(STEPS + 1):
        k = min(s * chunk, npix)
        idxs = order[:k]
        di = flat.copy()
        di[idxs] = blur[idxs]
        ii = blur.copy()
        ii[idxs] = flat[idxs]
        dbatch.append(di.reshape(SIZE, SIZE, 3))
        ibatch.append(ii.reshape(SIZE, SIZE, 3))
    dp = batch_prob(tf.constant(np.stack(dbatch), tf.float32)).numpy()
    ip = batch_prob(tf.constant(np.stack(ibatch), tf.float32)).numpy()
    dels.append(float(np.trapz(dp, dx=1.0 / STEPS)))
    inss.append(float(np.trapz(ip, dx=1.0 / STEPS)))
    dacc += dp
    iacc += ip
    cc += 1
    if (j + 1) % 50 == 0:
        print('  del/ins processed', j + 1, flush=True)
res.update(deletion_auc=round(float(np.mean(dels)), 4), insertion_auc=round(float(np.mean(inss)), 4), del_minus_ins=round(float(np.mean(inss) - np.mean(dels)), 4), deletion_curve=(dacc / cc).round(4).tolist(), insertion_curve=(iacc / cc).round(4).tolist(), n_delins=cc)
print('DELINS: del', res['deletion_auc'], 'ins', res['insertion_auc'], flush=True)
json.dump(res, open(os.path.join(SP, 'localization_results.json'), 'w'), indent=2)
print('SAVED', flush=True)
