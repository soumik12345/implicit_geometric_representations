import os
import wandb
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf
from keras import callbacks

from tqdm.autonotebook import tqdm


class ProgressBarCallback(callbacks.Callback):
    def __init__(self, epochs: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epochs = epochs
        self.p_bar = None

    def on_train_begin(self, logs=None):
        self.p_bar = tqdm(total=self.epochs)

    def on_epoch_end(self, epoch, logs=None):
        self.p_bar.update(n=1)
        self.p_bar.set_description(
            f"Training {epoch + 1}/{self.epochs}"
            if epoch + 1 < self.epochs
            else "Completed"
        )


class SDFVisualizationCallback(callbacks.Callback):
    def __init__(
        self,
        inputs: tf.Tensor,
        visualization_interval: int = 200,
        distance: float = 2,
        save_file: str = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.inputs = inputs
        self.visualization_interval = visualization_interval
        self.distance = distance
        self.save_file = save_file
        self.temp_file_path = "temp_plot.png"

    def visualize_sdf(self, epoch):
        """Reference: https://github.com/znah/notebooks/blob/master/tutorials/implicit_sdf.ipynb"""
        y, x = np.mgrid[
            -self.distance : self.distance : 256j, -self.distance : self.distance : 256j
        ]
        coordinates = np.stack([x, y], -1).astype(np.float32)
        sdf = self.model(coordinates)[..., 0]
        figure = plt.figure(figsize=(10, 8))
        plt.axis("equal")
        plt.grid()
        plt.contourf(x, y, sdf, 16)
        plt.colorbar()
        plt.contour(x, y, sdf, levels=[0.0], colors="white")
        if self.inputs is not None:
            _, sdf_gradients = self.model.compute_gradients(self.inputs)
            x, y = self.inputs.numpy().T
            u, v = sdf_gradients.numpy().T
            plt.quiver(x, y, u, v, color="white")
        plt.title(f"Signed Distance Field at Epoch: {epoch}")
        if self.save_file is not None:
            fig_1 = plt.gcf()
            fig_1.savefig(self.save_file)
        if wandb.run is not None:
            fig_1.savefig(self.temp_file_path)
            wandb.log(
                {"Signed Distance Field on Data": wandb.Image(self.temp_file_path)},
                step=epoch,
            )
        plt.show()

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.visualization_interval == 0:
            self.visualize_sdf(epoch + 1)
