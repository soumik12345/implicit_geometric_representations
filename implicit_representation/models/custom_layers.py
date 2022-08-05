import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class SoftPlus(layers.Layer):
    """Reference: https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html"""

    def __init__(self, beta: float = 1.0, threshold: float = 20.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.threshold = threshold

    def call(self, inputs):
        soft_plus_term = tf.math.log(1 + tf.math.exp(self.beta * inputs)) * (
            1 / self.beta
        )
        linear_term = inputs * self.beta
        return soft_plus_term + tf.math.maximum(linear_term, 0)
