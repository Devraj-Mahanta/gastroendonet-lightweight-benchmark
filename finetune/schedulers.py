import math
import tensorflow as tf

class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, initial_lr, warmup_steps, total_steps, min_lr=1e-07):
        super().__init__()
        self.initial_lr = float(initial_lr)
        self.warmup_steps = float(warmup_steps)
        self.total_steps = float(total_steps)
        self.min_lr = float(min_lr)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        w_step = tf.maximum(self.warmup_steps, 1.0)
        warmup_lr = self.initial_lr * step / w_step
        decay_steps = tf.maximum(self.total_steps - self.warmup_steps, 1.0)
        decay_step = step - self.warmup_steps
        cosine_lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1.0 + tf.cos(math.pi * decay_step / decay_steps))
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {'initial_lr': self.initial_lr, 'warmup_steps': self.warmup_steps, 'total_steps': self.total_steps, 'min_lr': self.min_lr}
