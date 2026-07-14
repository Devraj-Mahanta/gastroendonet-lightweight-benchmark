import sys, os
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)
import argparse, json
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import config as base_cfg
from finetune.config_finetune import ABLATION_CONFIGS, RESULTS_DIR, BASE_IMAGE_SIZE, HIGH_IMAGE_SIZE
from finetune.models import FT_REGISTRY
LAST_CONV_LAYERS = {'efficientnet_b1': ('efficientnetb1', 'top_activation'), 'mobilenetv3_small': ('MobileNetV3Small', 'multiply_1')}

def _get_gradcam_model(model, model_name):
    (backbone_name, layer_name) = LAST_CONV_LAYERS[model_name]
    backbone = None
    for layer in model.layers:
        if layer.name.lower() == backbone_name.lower():
            backbone = layer
            break
    if backbone is None:
        for layer in model.layers:
            if backbone_name.lower() in layer.name.lower():
                backbone = layer
                break
    if backbone is None:
        raise ValueError(f"Backbone layer '{backbone_name}' not found in model. Available layers: {[l.name for l in model.layers]}")
    target_layer = None
    for layer in backbone.layers:
        if layer.name == layer_name:
            target_layer = layer
            break
    if target_layer is None:
        for layer in reversed(backbone.layers):
            try:
                out_shape = layer.output_shape
                if isinstance(out_shape, (list, tuple)) and len(out_shape) == 4:
                    target_layer = layer
                    break
            except Exception:
                pass
    if target_layer is None:
        raise ValueError(f"Could not find a suitable conv layer in backbone '{backbone_name}'.")
    grad_model = tf.keras.Model(inputs=model.inputs, outputs=[target_layer.output, model.output])
    return grad_model

def compute_gradcam(grad_model, img_array, class_idx):
    inp = tf.cast(tf.expand_dims(img_array, 0), tf.float32)
    with tf.GradientTape() as tape:
        (conv_out, preds) = grad_model(inp)
        tape.watch(conv_out)
        class_score = preds[:, class_idx]
    grads = tape.gradient(class_score, conv_out)
    pooled_grad = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heatmap = conv_out @ pooled_grad[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0.0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-08)
    return (heatmap.numpy(), preds.numpy()[0])

def overlay_heatmap(img, heatmap, alpha=0.45, colormap='jet'):
    heatmap_resized = np.array(tf.image.resize(heatmap[..., np.newaxis], [img.shape[0], img.shape[1]])[..., 0])
    colour_map = cm.get_cmap(colormap)
    coloured = colour_map(heatmap_resized)[..., :3]
    overlay = (1 - alpha) * img + alpha * coloured
    return np.clip(overlay, 0.0, 1.0)

def _load_test_samples(image_size, n_per_class=4, seed=42):
    with open(base_cfg.SPLIT_FILE) as f:
        split = json.load(f)
    paths = split['test']['paths']
    labels = split['test']['labels']
    rng = np.random.default_rng(seed)
    by_cls = {i: [] for i in range(base_cfg.NUM_CLASSES)}
    for (p, l) in zip(paths, labels):
        by_cls[l].append(p)
    samples = {}
    for (cls_idx, cls_name) in enumerate(base_cfg.CLASS_NAMES):
        chosen = rng.choice(by_cls[cls_idx], size=min(n_per_class, len(by_cls[cls_idx])), replace=False)
        imgs = []
        for path in chosen:
            raw = tf.io.read_file(path)
            img = tf.image.decode_image(raw, channels=3, expand_animations=False)
            img.set_shape([None, None, 3])
            img = tf.image.resize(img, [image_size, image_size])
            img = tf.cast(img, tf.float32) / 255.0
            imgs.append((img.numpy(), cls_idx))
        samples[cls_name] = imgs
    return samples

def generate_gradcam_grid(model, grad_model, model_name, image_size, n_per_class=4, seed=42):
    samples = _load_test_samples(image_size, n_per_class, seed)
    n_classes = base_cfg.NUM_CLASSES
    fig_w = 3.5 * n_per_class
    fig_h = 3.5 * n_classes
    (fig, axes) = plt.subplots(n_classes, n_per_class, figsize=(fig_w, fig_h))
    fig.suptitle(f'Grad-CAM — {model_name}', fontsize=14, y=1.01)
    for (row, cls_name) in enumerate(base_cfg.CLASS_NAMES):
        imgs = samples[cls_name]
        for col in range(n_per_class):
            ax = axes[row][col] if n_per_class > 1 else axes[row]
            ax.axis('off')
            if col >= len(imgs):
                continue
            (img_array, true_label) = imgs[col]
            pred_class = int(np.argmax(model(tf.expand_dims(img_array, 0), training=False).numpy()))
            (heatmap, probs) = compute_gradcam(grad_model, img_array, pred_class)
            overlaid = overlay_heatmap(img_array, heatmap)
            ax.imshow(overlaid)
            pred_name = base_cfg.CLASS_NAMES[pred_class]
            colour = 'green' if pred_class == true_label else 'red'
            ax.set_title(f'True: {cls_name}\nPred: {pred_name} ({probs[pred_class]:.2f})', fontsize=8, color=colour)
        axes[row][0].set_ylabel(cls_name, fontsize=10, rotation=90, labelpad=4)
    plt.tight_layout()
    return fig

def _load_ft_model_for_gradcam(model_name, cfg_name, run_id):
    from finetune.train_finetune import _load_weights_by_name
    ft_cfg = ABLATION_CONFIGS[cfg_name]
    img_size = HIGH_IMAGE_SIZE[model_name] if ft_cfg['image_size_key'] == 'high' else BASE_IMAGE_SIZE[model_name]
    tf.keras.backend.clear_session()
    run_seed = base_cfg.SEED + (run_id - 1)
    (model, _) = FT_REGISTRY[model_name](run_seed=run_seed, image_size=img_size, rich_head=ft_cfg['rich_head'], backbone_weights='imagenet')
    model.trainable = True
    from finetune.config_finetune import CHECKPOINTS_DIR
    ckpt = os.path.join(CHECKPOINTS_DIR, f'{model_name}_{cfg_name}_run{run_id}_best.h5')
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt}\nTrain the model first (train_finetune.py).')
    try:
        model.load_weights(ckpt)
    except Exception:
        _load_weights_by_name(model, ckpt)
    return (model, img_size)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='efficientnet_b1', choices=list(FT_REGISTRY.keys()))
    p.add_argument('--config', default='full', choices=list(ABLATION_CONFIGS.keys()))
    p.add_argument('--run', type=int, default=1)
    p.add_argument('--n_per_class', type=int, default=4, help='Number of test images per class in the grid')
    return p.parse_args()

def main():
    args = parse_args()
    print(f'Generating Grad-CAM for {args.model}  cfg={args.config}  run={args.run}')
    (model, image_size) = _load_ft_model_for_gradcam(args.model, args.config, args.run)
    grad_model = _get_gradcam_model(model, args.model)
    fig = generate_gradcam_grid(model, grad_model, args.model, image_size, n_per_class=args.n_per_class)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f'gradcam_{args.model}_{args.config}_run{args.run}.png')
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'Grad-CAM grid saved -> {out}')
if __name__ == '__main__':
    main()
