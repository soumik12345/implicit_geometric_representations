from typing import Callable

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, initializers, activations


class SDFModelBase(keras.Model):
    def __init__(
        self,
        num_points: int,
        units: int = 80,
        num_intermediate_layers=2,
        activation: Callable = activations.softplus,
        point_loss_coeff: float = 100.0,
        eikonal_coefficient: float = 1.0,
        num_padding_points: int = 500,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.num_points = num_points
        self.units = units
        self.num_intermediate_layers = num_intermediate_layers
        self.point_loss_coeff = point_loss_coeff
        self.eikonal_coefficient = eikonal_coefficient
        self.num_padding_points = num_padding_points
        self.num_dimensions = 2
        self.activation = activation
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
            eikonal_loss = self.eikonal_coefficient * tf.reduce_mean(
                tf.square(normalized_sdf_gradients - 1.0)
            )
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
