import numpy as np
import tensorflow as tf

from .models import SDFModelBase


class SDFRenderer:
    """Reference: https://github.com/znah/notebooks/blob/master/tutorials/implicit_sdf.ipynb"""

    def __init__(
        self, frame_size: int, field_of_view: float = 0.7, iterations: int = 50
    ) -> None:
        self.frame_size = frame_size
        self.field_of_view = field_of_view
        self.iterations = iterations

    @tf.function(jit_compile=True)
    def render(
        self, sdf_model: SDFModelBase, transform_matrix: tf.Tensor, offset: float
    ):
        grid_range = tf.linspace(
            -self.field_of_view, self.field_of_view, self.field_of_view
        )
        x, y = tf.meshgrid(grid_range, -grid_range)
        rays = tf.stack([x, y, -tf.ones_like(x)], -1)
        normalized_rays = tf.nn.l2_normalize(rays, -1)
        transformed_rays = tf.matmul(normalized_rays, transform_matrix)
        position = tf.matmul(np.float32([[0, 0, 2.5]]), transform_matrix)
        position = position + tf.zeros_like(transformed_rays)
        for _ in tf.range(self.iterations):
            sdf = sdf_model(position)
            position += transformed_rays * (sdf - offset)
        _, gradients = sdf_model.compute_gradients(position)
        return gradients * 0.5 + 0.5
