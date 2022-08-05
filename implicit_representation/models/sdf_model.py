import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, initializers, activations


class SDFModelBase(keras.Model):
    def __init__(
        self,
        num_points: int,
        units: int = 80,
        num_intermediate_layers=2,
        point_loss_coeff: float = 100.0,
        num_padding_points: int = 500,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.num_points = num_points
        self.units = units
        self.num_intermediate_layers = num_intermediate_layers
        self.point_loss_coeff = point_loss_coeff
        self.num_padding_points = num_padding_points
        self.num_dimensions = 2
        self.activation = activations.softplus
        self._build_layers()

    def _build_layers(self):
        self.initial_layer = layers.Dense(
            self.units,
            self.activation,
            kernel_initializer=initializers.RandomNormal(0.0, 3.0),
        )
        self.intermedieate_layers = [
            layers.Dense(self.units, self.activation)
            for _ in range(self.num_intermediate_layers)
        ]
        self.output_layer = layers.Dense(
            1,
            kernel_initializer=initializers.RandomNormal(0.01, 0.001),
            bias_initializer=initializers.Constant(-0.2),
        )

    def call(self, inputs):
        x = self.initial_layer(inputs)
        for layer in self.intermedieate_layers:
            x = layer(x)
        return self.output_layer(x)

    def compute_gradients(self, inputs):
        with tf.GradientTape() as grad_tape:
            grad_tape.watch(inputs)
            sdf = self(inputs)
        sdf_gradients = grad_tape.gradient(sdf, inputs)
        return sdf, sdf_gradients

    def pad_inputs(self, inputs):
        inputs = inputs[0]
        num_points = tf.shape(inputs)[0]
        padded_points = tf.concat(
            [
                inputs,
                tf.random.uniform(
                    [self.num_padding_points, self.num_dimensions], -2.0, 2.0
                ),
            ],
            0,
        )
        return padded_points

    def compute_point_loss(self, sdf):
        point_loss_term = tf.reduce_mean(tf.square(sdf[: self.num_points]))
        padded_point_loss_term = tf.reduce_mean(
            tf.exp(-tf.square(sdf[self.num_points :] / 0.02))
        )
        total_point_loss = (
            point_loss_term * self.point_loss_coeff + padded_point_loss_term
        )
        return point_loss_term, padded_point_loss_term, total_point_loss

    def train_step(self, inputs):
        with tf.GradientTape() as train_tape:
            padded_inputs = self.pad_inputs(inputs)
            sdf, sdf_gradients = self.compute_gradients(padded_inputs)
            (
                point_loss_term,
                padded_point_loss_term,
                total_point_loss,
            ) = self.compute_point_loss(sdf)
            normalized_sdf_gradients = tf.reduce_sum(tf.square(sdf_gradients), -1)
            eikonal_loss = tf.reduce_mean(tf.square(normalized_sdf_gradients - 1.0))
            loss = total_point_loss + eikonal_loss
        trainable_variables = train_tape.watched_variables()
        gradients = train_tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        return {
            "loss": loss,
            "point_loss_term": point_loss_term,
            "padded_point_loss_term": padded_point_loss_term,
            "total_point_loss": total_point_loss,
            "eikonal_loss": eikonal_loss,
        }

    def visualize(self, inputs=None, distance=2, save_file: str = None):
        """Reference: https://github.com/znah/notebooks/blob/master/tutorials/implicit_sdf.ipynb"""
        y, x = np.mgrid[-distance:distance:256j, -distance:distance:256j]
        coords = np.stack([x, y], -1).astype(np.float32)
        sdf = self(coords)[..., 0]
        f = plt.figure(figsize=(10, 8))
        plt.axis("equal")
        plt.grid()
        plt.contourf(x, y, sdf, 16)
        plt.colorbar()
        plt.contour(x, y, sdf, levels=[0.0], colors="white")
        if inputs is not None:
            _, sdf_gradients = self.compute_gradients(inputs)
            x, y = inputs.numpy().T
            u, v = sdf_gradients.numpy().T
            plt.quiver(x, y, u, v, color="white")
        plt.show()
        if save_file is not None:
            plt.savefig(save_file)
