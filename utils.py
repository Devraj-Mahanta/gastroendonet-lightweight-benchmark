import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import config as cfg

def plot_run_history(phase_histories: list, model_name: str, run_id: int):
    (fig, axes) = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(f'{model_name}  —  Run {run_id}', fontsize=12)
    phase_colors = ['#4C72B0', '#DD8452', '#55A868']
    phase_labels = ['Phase 1 (head)', 'Phase 2 (partial)', 'Phase 3 (full)']
    offset = 0
    for (ax_acc, ax_loss) in [(axes[0], axes[1])]:
        offset = 0
        for (ph_i, hist) in enumerate(phase_histories):
            n_ep = len(hist['accuracy'])
            x = list(range(offset + 1, offset + n_ep + 1))
            color = phase_colors[ph_i]
            label = phase_labels[ph_i]
            ax_acc.plot(x, hist['accuracy'], color=color, ls='-', label=f'{label} train')
            ax_acc.plot(x, hist['val_accuracy'], color=color, ls='--', label=f'{label} val')
            ax_loss.plot(x, hist['loss'], color=color, ls='-', label=f'{label} train')
            ax_loss.plot(x, hist['val_loss'], color=color, ls='--', label=f'{label} val')
            if ph_i < len(phase_histories) - 1:
                boundary = offset + n_ep
                ax_acc.axvline(boundary, color='gray', ls=':', lw=0.8)
                ax_loss.axvline(boundary, color='gray', ls=':', lw=0.8)
            offset += n_ep
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Epoch (global)')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend(fontsize=6, ncol=2)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_title('Loss')
    axes[1].set_xlabel('Epoch (global)')
    axes[1].set_ylabel('Loss')
    axes[1].legend(fontsize=6, ncol=2)
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    out = os.path.join(cfg.RESULTS_DIR, f'{model_name}_run{run_id}_history.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'[plot] history -> {out}')

def plot_confusion_matrix(model, dataset, model_name: str, run_id: int):
    (all_pred, all_true) = ([], [])
    for (imgs, labels) in dataset:
        probs = model(imgs, training=False)
        all_pred.append(tf.argmax(probs, axis=-1).numpy())
        all_true.append(tf.argmax(labels, axis=-1).numpy())
    y_pred = np.concatenate(all_pred)
    y_true = np.concatenate(all_true)
    cm = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES), dtype=int)
    for (t, p) in zip(y_true, y_pred):
        cm[t, p] += 1
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-08)
    (fig, ax) = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(cfg.NUM_CLASSES))
    ax.set_yticks(range(cfg.NUM_CLASSES))
    ax.set_xticklabels(cfg.CLASS_NAMES, rotation=40, ha='right', fontsize=9)
    ax.set_yticklabels(cfg.CLASS_NAMES, fontsize=9)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'{model_name} — Run {run_id} — Confusion Matrix')
    for i in range(cfg.NUM_CLASSES):
        for j in range(cfg.NUM_CLASSES):
            ax.text(j, i, f'{cm[i, j]}\n({cm_norm[i, j]:.2f})', ha='center', va='center', color='white' if cm_norm[i, j] > 0.5 else 'black', fontsize=8)
    plt.tight_layout()
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    out = os.path.join(cfg.RESULTS_DIR, f'{model_name}_run{run_id}_confusion.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'[plot] confusion -> {out}')
