import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from abc import abstractmethod, ABC
from PIL import Image, ImageFont, ImageDraw


class PointCloud2D(ABC):
    def __init__(self, sample_percentage: float):
        self.points = None
        self.sample_percentage = sample_percentage

    @abstractmethod
    def build(self) -> tf.Tensor:
        pass

    def __len__(self):
        return len(self.points)

    @abstractmethod
    def create_points(self) -> None:
        pass

    def preprocess_points(self) -> None:
        self.points = self.points.astype(np.float32)
        self.points -= self.points.mean(0)
        self.points /= self.points.std() * 2.0

    def sample_points(self) -> None:
        self.points = self.points[
            np.random.rand(len(self.points)) < self.sample_percentage
        ]

    def plot_points(self, save_file: str = None, transpose: bool = True) -> None:
        x, y = self.points.T if transpose else self.points
        plt.clf()
        plt.plot(x, y, ".")
        plt.axis("equal")
        plt.show()
        if save_file is not None:
            plt.savefig(save_file)


class PointCloud2DFromFont(PointCloud2D):
    def __init__(self, sample_percentage, font_file: str, font_size: int):
        super().__init__(sample_percentage)
        self.font = ImageFont.truetype(font_file, font_size)

    def build(self, query: str, padding: int) -> tf.Tensor:
        self.create_points(query, padding)
        self.preprocess_points()
        self.sample_points()
        return tf.convert_to_tensor(self.points)

    def create_points(self, query: str, padding: int) -> None:
        w, h = self.font.getsize(query)
        image = Image.new("L", (w + padding * 2, h + padding * 2))
        draw = ImageDraw.Draw(image)
        draw.text(
            (padding, padding), query, font=self.font, stroke_width=1, stroke_fill=255
        )
        image = np.float32(image) / 255.0
        y, x = image.nonzero()
        self.points = np.stack([x, -y], -1).astype(np.float32)


class PointCloud2DBatman(PointCloud2D):
    def __init__(self, sample_percentage):
        super().__init__(sample_percentage)

    def build(self) -> tf.Tensor:
        self.create_points()
        self.preprocess_points()
        self.sample_points()
        return tf.convert_to_tensor(self.points)

    def create_points(self):
        Y = np.arange(-4, 4, 0.005)
        X = np.zeros((0))
        for y in Y:
            X = np.append(
                X,
                abs(y / 2)
                - 0.09137 * y**2
                + math.sqrt(1 - (abs(abs(y) - 2) - 1) ** 2)
                - 3,
            )
        Y1 = np.append(np.arange(-7, -3, 0.01), np.arange(3, 7, 0.01))
        X1 = np.zeros((0))
        for y in Y1:
            X1 = np.append(X1, 3 * math.sqrt(-((y / 7) ** 2) + 1))
        X = np.append(X, X1)
        Y = np.append(Y, Y1)
        Y1 = np.append(np.arange(-7.0, -4, 0.01), np.arange(4, 7.01, 0.01))
        X1 = np.zeros((0))
        for y in Y1:
            X1 = np.append(X1, -3 * math.sqrt(-((y / 7) ** 2) + 1))
        X = np.append(X, X1)
        Y = np.append(Y, Y1)
        Y1 = np.append(np.arange(-1, -0.8, 0.01), np.arange(0.8, 1, 0.01))
        X1 = np.zeros((0))
        for y in Y1:
            X1 = np.append(X1, 9 - 8 * abs(y))
        X = np.append(X, X1)
        Y = np.append(Y, Y1)
        Y1 = np.arange(-0.5, 0.5, 0.05)
        X1 = np.zeros((0))
        for y in Y1:
            X1 = np.append(X1, 2)
        X = np.append(X, X1)
        Y = np.append(Y, Y1)
        Y1 = np.append(np.arange(-2.9, -1, 0.01), np.arange(1, 2.9, 0.01))
        X1 = np.zeros((0))
        for y in Y1:
            X1 = np.append(
                X1,
                1.5 - 0.5 * abs(y) - 1.89736 * (math.sqrt(3 - y**2 + 2 * abs(y)) - 2),
            )
        X = np.append(X, X1)
        Y = np.append(Y, Y1)
        Y1 = np.append(np.arange(-0.7, -0.45, 0.01), np.arange(0.45, 0.7, 0.01))
        X1 = np.zeros((0))
        for y in Y1:
            X1 = np.append(X1, 3 * abs(y) + 0.75)
        X = np.append(X, X1)
        Y = np.append(Y, Y1)
        self.points = np.stack([Y, X], -1).astype(np.float32)
        print(self.points.shape)
