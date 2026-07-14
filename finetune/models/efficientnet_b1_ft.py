import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import tensorflow as tf
import config as cfg

def get_model(run_seed=cfg.SEED, image_size=240, rich_head=False, backbone_weights='imagenet', weight_decay=0.0):
    tf.random.set_seed(run_seed)
    reg = tf.keras.regularizers.l2(weight_decay) if weight_decay > 0 else None
    inputs = tf.keras.Input(shape=(image_size, image_size, cfg.CHANNELS), name='input')
    x = tf.keras.layers.Rescaling(255.0, name='rescale_to_0_255')(inputs)
    base = tf.keras.applications.EfficientNetB1(include_top=False, weights=backbone_weights, input_shape=(image_size, image_size, cfg.CHANNELS))
    x = base(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name='gap')(x)
    if rich_head:
        x = tf.keras.layers.BatchNormalization(name='head_bn')(x)
        x = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_normal', kernel_regularizer=reg, name='head_fc1')(x)
        x = tf.keras.layers.Dropout(0.15, name='head_drop1')(x)
        x = tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_normal', kernel_regularizer=reg, name='head_fc2')(x)
        x = tf.keras.layers.Dropout(0.1, name='head_drop2')(x)
    else:
        x = tf.keras.layers.Dropout(0.2, name='drop')(x)
    outputs = tf.keras.layers.Dense(cfg.NUM_CLASSES, activation='softmax', kernel_regularizer=reg, name='classifier')(x)
    model = tf.keras.Model(inputs, outputs, name='EfficientNet-B1-FT')
    base.trainable = False
    return (model, base)
