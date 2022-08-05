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
            if epoch + 1 == self.epochs
            else "Completed"
        )
