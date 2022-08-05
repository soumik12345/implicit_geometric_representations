import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import callbacks

from tqdm.autonotebook import tqdm


class ProgressBarCallback(callbacks.Callback):
    def __init__(self, epochs: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epochs = epochs
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

    def visualize_sdf(self):
        """Reference: https://github.com/znah/notebooks/blob/master/tutorials/implicit_sdf.ipynb"""
        y, x = np.mgrid[
            -self.distance : self.distance : 256j, -self.distance : self.distance : 256j
        ]
        coordinates = np.stack([x, y], -1).astype(np.float32)
        sdf = self.model(coordinates)[..., 0]
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
        if self.save_file is not None:
            plt.savefig(self.save_file)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.visualization_interval:
            self.visualize_sdf()
