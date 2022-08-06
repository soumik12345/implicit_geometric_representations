import os
import wandb
import trimesh
import numpy as np
import tensorflow as tf
from plotly import graph_objects as go


# def _random_choice(inputs, n_samples):
#     uniform_log_prob = tf.expand_dims(tf.zeros(tf.shape(inputs)[0]), 0)
#     indices = tf.random.categorical(uniform_log_prob, n_samples)
#     indices = tf.squeeze(indices, 0, name="random_choice_ind")
#     return tf.gather(inputs, indices)


class PointCloudFromModel:
    def __init__(self, artifact_address: str):
        self.artifact_address = artifact_address
        self.model_path = None

    def build_dataset(self):
        self.mesh = trimesh.load(self.model_path)
        vertices = np.float32(self.mesh.vertices)
        vertices -= vertices.mean(0)
        self.point_cloud = vertices / vertices.ptp() * 0.4
        return self.point_cloud

    def fetch_dataset(self):
        if wandb.run is None:
            api = wandb.Api()
            artifact = api.artifact(self.artifact_address, type="dataset")
        else:
            artifact = wandb.use_artifact(self.artifact_address, type="dataset")
        artifact_dir = artifact.download()
        return artifact_dir

    def visualize(self, num_points: int):
        point_cloud = self.point_cloud[
            np.random.choice(len(self.point_cloud), num_points, False)
        ]
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=point_cloud[:, 0],
                    y=point_cloud[:, 1],
                    z=point_cloud[:, 2],
                    mode="markers",
                    marker=dict(
                        size=6,
                        color=point_cloud[:, 2],
                        colorscale="Viridis",
                        opacity=0.6,
                    ),
                )
            ]
        )
        fig.show()


class BunnyPointCloud(PointCloudFromModel):
    def __init__(self, artifact_address: str):
        super().__init__(artifact_address)
        self.model_path = os.path.join(
            self.fetch_dataset(), "reconstruction", "bun_zipper.ply"
        )
