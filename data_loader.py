import os
import json
import math
import numpy as np
import tensorflow as tf
import config as cfg

def generate_and_save_split():
    import random
    (paths, labels) = ([], [])
    for cls in cfg.CLASS_NAMES:
        cls_dir = os.path.join(cfg.DATA_DIR, cls)
        for fname in sorted(os.listdir(cls_dir)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                paths.append(os.path.join(cls_dir, fname).replace('\\', '/'))
                labels.append(cfg.CLASS_NAMES.index(cls))
    n = len(paths)
    rng = random.Random(cfg.SEED)
    idx = list(range(n))
    rng.shuffle(idx)
    paths = [paths[i] for i in idx]
    labels = [labels[i] for i in idx]
    n_train = int(n * cfg.TRAIN_SPLIT)
    n_val = int(n * cfg.VAL_SPLIT)
    split = {'train': {'paths': paths[:n_train], 'labels': labels[:n_train]}, 'val': {'paths': paths[n_train:n_train + n_val], 'labels': labels[n_train:n_train + n_val]}, 'test': {'paths': paths[n_train + n_val:], 'labels': labels[n_train + n_val:]}, 'seed': cfg.SEED, 'total': n}
    os.makedirs(os.path.dirname(cfg.SPLIT_FILE), exist_ok=True)
    with open(cfg.SPLIT_FILE, 'w') as f:
        json.dump(split, f, indent=2)
    print(f'[split] Saved to {cfg.SPLIT_FILE}')
    print(f'[split] train={n_train}  val={n_val}  test={n - n_train - n_val}')
    return split

def load_split():
    if not os.path.exists(cfg.SPLIT_FILE):
        raise FileNotFoundError(f'Split file not found: {cfg.SPLIT_FILE}\nRun:  python generate_split.py')
    with open(cfg.SPLIT_FILE) as f:
        split = json.load(f)
    return split

def _parse_image(path, label):
    raw = tf.io.read_file(path)
    img = tf.image.decode_image(raw, channels=cfg.CHANNELS, expand_animations=False)
    img.set_shape([None, None, cfg.CHANNELS])
    img = tf.image.resize(img, [cfg.IMAGE_SIZE, cfg.IMAGE_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    return (img, label)

def _colour_jitter(img, label):
    img = tf.image.random_brightness(img, max_delta=cfg.AUG_BRIGHTNESS)
    img = tf.image.random_contrast(img, 1.0 - cfg.AUG_CONTRAST, 1.0 + cfg.AUG_CONTRAST)
    img = tf.image.random_saturation(img, 1.0 - cfg.AUG_SATURATION, 1.0 + cfg.AUG_SATURATION)
    img = tf.image.random_hue(img, max_delta=0.05)
    img = tf.clip_by_value(img, 0.0, 1.0)
    return (img, label)
_spatial_aug = tf.keras.Sequential([tf.keras.layers.RandomFlip('horizontal'), tf.keras.layers.RandomRotation(factor=cfg.AUG_ROTATION), tf.keras.layers.RandomZoom(height_factor=(-cfg.AUG_ZOOM, cfg.AUG_ZOOM), width_factor=(-cfg.AUG_ZOOM, cfg.AUG_ZOOM))], name='spatial_augmentation')

def _apply_spatial(imgs, labels):
    return (_spatial_aug(imgs, training=True), labels)

def _one_hot(img, label):
    return (img, tf.one_hot(label, cfg.NUM_CLASSES))

def compute_class_weights(labels):
    counts = np.bincount(labels, minlength=cfg.NUM_CLASSES).astype(np.float32)
    total = counts.sum()
    weights = total / (cfg.NUM_CLASSES * counts)
    return {i: float(w) for (i, w) in enumerate(weights)}

def build_datasets():
    split = load_split()
    train_paths = split['train']['paths']
    train_labels = split['train']['labels']
    val_paths = split['val']['paths']
    val_labels = split['val']['labels']
    test_paths = split['test']['paths']
    test_labels = split['test']['labels']
    class_weights = compute_class_weights(train_labels)

    def _make(paths, labels, augment=False):
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if augment:
            ds = ds.shuffle(len(paths), seed=cfg.SEED, reshuffle_each_iteration=True)
        ds = ds.map(_parse_image, num_parallel_calls=tf.data.AUTOTUNE)
        if augment:
            ds = ds.map(_colour_jitter, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(cfg.BATCH_SIZE)
        if augment:
            ds = ds.map(_apply_spatial, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(_one_hot, num_parallel_calls=tf.data.AUTOTUNE)
        return ds.prefetch(tf.data.AUTOTUNE)
    train_ds = _make(train_paths, train_labels, augment=True)
    val_ds = _make(val_paths, val_labels, augment=False)
    test_ds = _make(test_paths, test_labels, augment=False)
    print(f'[data] train={len(train_paths)}  val={len(val_paths)}  test={len(test_paths)}')
    cw_str = '  '.join((f'{cfg.CLASS_NAMES[i]}={v:.3f}' for (i, v) in class_weights.items()))
    print(f'[data] class weights: {cw_str}')
    return (train_ds, val_ds, test_ds, len(train_paths), len(val_paths), class_weights)
