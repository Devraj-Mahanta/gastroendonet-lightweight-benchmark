import tensorflow as tf

class CategoricalFocalLoss(tf.keras.losses.Loss):

    def __init__(self, gamma=2.0, label_smoothing=0.05, name='categorical_focal_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)

    def call(self, y_true, y_pred):
        if self.label_smoothing > 0.0:
            n_cls = tf.cast(tf.shape(y_true)[-1], tf.float32)
            y_true = y_true * (1.0 - self.label_smoothing) + self.label_smoothing / n_cls
        y_pred = tf.clip_by_value(y_pred, 1e-07, 1.0 - 1e-07)
        ce = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
        focal_weight = tf.pow(1.0 - p_t, self.gamma)
        return focal_weight * ce

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'gamma': self.gamma, 'label_smoothing': self.label_smoothing})
        return cfg
