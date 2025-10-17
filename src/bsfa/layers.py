from tensorflow.keras import layers
import tensorflow as tf

class AccountingConstraintInOriginalSpace(layers.Layer):
    """Enforce total_assets = total_liabilities + total_equity in original space.
    First unscale, adjust equity, then rescale back.
    """
    def __init__(self, y_mu, y_sigma, **kwargs):
        super().__init__(**kwargs)
        self.y_mu = tf.constant(y_mu, dtype=tf.float32)
        self.y_sigma = tf.constant(y_sigma, dtype=tf.float32)

    def call(self, inputs):
        y = inputs * self.y_sigma + self.y_mu  # (B,F)
        assets = tf.maximum(y[:, 0:1], 0.0)
        liabilities = tf.maximum(y[:, 1:2], 0.0)
        equity = assets - liabilities
        y_adj = tf.concat([assets, liabilities, equity, y[:, 3:]], axis=1)
        return (y_adj - self.y_mu) / self.y_sigma
