import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import numpy as np
import tensorflow as tf
import config as base_cfg

def _parse_image(path, label, image_size):
    raw = tf.io.read_file(path)
    img = tf.image.decode_image(raw, channels=base_cfg.CHANNELS, expand_animations=False)
    img.set_shape([None, None, base_cfg.CHANNELS])
    img = tf.image.resize(img, [image_size, image_size])
    img = tf.cast(img, tf.float32) / 255.0
    return (img, label)

def _colour_jitter(img, label, brightness, contrast, saturation):
    img = tf.image.random_brightness(img, max_delta=brightness)
    img = tf.image.random_contrast(img, 1.0 - contrast, 1.0 + contrast)
    img = tf.image.random_saturation(img, 1.0 - saturation, 1.0 + saturation)
    img = tf.image.random_hue(img, max_delta=0.05)
    img = tf.clip_by_value(img, 0.0, 1.0)
    return (img, label)

def _make_colour_fn(aug_params):
    b = aug_params['brightness']
    c = aug_params['contrast']
    s = aug_params['saturation']
    return lambda img, lbl: _colour_jitter(img, lbl, b, c, s)

def _make_spatial_aug(aug_params):
    flip = tf.keras.layers.RandomFlip('horizontal_and_vertical')
    rot = tf.keras.layers.RandomRotation(factor=aug_params['rotation'])
    zoom = tf.keras.layers.RandomZoom(height_factor=(-aug_params['zoom'], aug_params['zoom']), width_factor=(-aug_params['zoom'], aug_params['zoom']))
    seq = tf.keras.Sequential([flip, rot, zoom], name='spatial_aug_ft')
    return seq

def _apply_mixup(images, labels, alpha):

    def _fn(imgs, labs):
        lam = np.float32(np.random.beta(alpha, alpha))
        idx = np.random.permutation(imgs.shape[0])
        mixed = lam * imgs + (1.0 - lam) * imgs[idx]
        lmix = lam * labs + (1.0 - lam) * labs[idx]
        return (mixed.astype(np.float32), lmix.astype(np.float32))
    (imgs_out, labs_out) = tf.py_function(lambda x, y: _fn(x.numpy(), y.numpy()), [images, labels], [tf.float32, tf.float32])
    imgs_out.set_shape(images.shape)
    labs_out.set_shape(labels.shape)
    return (imgs_out, labs_out)

def _apply_cutmix(images, labels, alpha):

    def _fn(imgs, labs):
        (B, H, W, C) = imgs.shape
        lam = np.float32(np.random.beta(alpha, alpha))
        idx = np.random.permutation(B)
        cut = np.sqrt(1.0 - lam)
        ch = int(H * cut)
        cw = int(W * cut)
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        x1 = max(cx - cw // 2, 0)
        x2 = min(cx + cw // 2, W)
        y1 = max(cy - ch // 2, 0)
        y2 = min(cy + ch // 2, H)
        imgs_m = imgs.copy()
        imgs_m[:, y1:y2, x1:x2, :] = imgs[idx, y1:y2, x1:x2, :]
        lam_a = 1.0 - float((y2 - y1) * (x2 - x1)) / (H * W)
        labs_m = lam_a * labs + (1.0 - lam_a) * labs[idx]
        return (imgs_m.astype(np.float32), labs_m.astype(np.float32))
    (imgs_out, labs_out) = tf.py_function(lambda x, y: _fn(x.numpy(), y.numpy()), [images, labels], [tf.float32, tf.float32])
    imgs_out.set_shape(images.shape)
    labs_out.set_shape(labels.shape)
    return (imgs_out, labs_out)

def _one_hot(img, label):
    return (img, tf.one_hot(label, base_cfg.NUM_CLASSES))

def build_datasets_ft(image_size=224, batch_size=32, mixup_alpha=0.0, cutmix_alpha=0.0, strong_aug=False):
    from finetune.config_finetune import STD_AUG, STRONG_AUG
    aug_params = STRONG_AUG if strong_aug else STD_AUG
    colour_fn = _make_colour_fn(aug_params)
    spatial_fn = _make_spatial_aug(aug_params)
    with open(base_cfg.SPLIT_FILE) as f:
        split = json.load(f)
    train_paths = split['train']['paths']
    train_labels = split['train']['labels']
    val_paths = split['val']['paths']
    val_labels = split['val']['labels']
    test_paths = split['test']['paths']
    test_labels = split['test']['labels']
    counts = np.bincount(train_labels, minlength=base_cfg.NUM_CLASSES).astype(np.float32)
    class_weights = {i: float(counts.sum() / (base_cfg.NUM_CLASSES * c)) for (i, c) in enumerate(counts)}

    def _make_train(paths, labels):
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.shuffle(len(paths), seed=base_cfg.SEED, reshuffle_each_iteration=True)
        ds = ds.map(lambda p, l: _parse_image(p, l, image_size), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(colour_fn, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size, drop_remainder=True)
        ds = ds.map(lambda x, y: (spatial_fn(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(_one_hot, num_parallel_calls=tf.data.AUTOTUNE)
        if mixup_alpha > 0:
            ds = ds.map(lambda x, y: _apply_mixup(x, y, mixup_alpha), num_parallel_calls=tf.data.AUTOTUNE)
        if cutmix_alpha > 0:
            ds = ds.map(lambda x, y: _apply_cutmix(x, y, cutmix_alpha), num_parallel_calls=tf.data.AUTOTUNE)
        return ds.prefetch(tf.data.AUTOTUNE)

    def _make_eval(paths, labels):
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.map(lambda p, l: _parse_image(p, l, image_size), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size)
        ds = ds.map(_one_hot, num_parallel_calls=tf.data.AUTOTUNE)
        return ds.prefetch(tf.data.AUTOTUNE)
    train_ds = _make_train(train_paths, train_labels)
    val_ds = _make_eval(val_paths, val_labels)
    test_ds = _make_eval(test_paths, test_labels)
    aug_str = f'mixup={mixup_alpha} cutmix={cutmix_alpha} strong={strong_aug}'
    print(f'[ft-data] size={image_size} batch={batch_size} {aug_str}')
    print(f'[ft-data] train={len(train_paths)} val={len(val_paths)} test={len(test_paths)}')
    return (train_ds, val_ds, test_ds, len(train_paths), len(val_paths), class_weights)
