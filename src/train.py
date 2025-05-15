import tensorflow as tf
import numpy as np
from verify_tfrecord import load_tfrecord_dataset  # replace with your actual module

# Constants
BATCH_SIZE = 32
SHUFFLE_BUFFER = 1000
MAX_LEN = 80  # tensor depth


DATASET_SIZE = 2  # replace with your actual dataset size

def split_dataset(dataset, dataset_size, train_frac=0.8, val_frac=0.1):
    train_size = int(train_frac * dataset_size)
    val_size = int(val_frac * dataset_size)

    train_ds = dataset.take(train_size)
    val_ds = dataset.skip(train_size).take(val_size)
    test_ds = dataset.skip(train_size + val_size)

    return train_ds, val_ds, test_ds
class Model(tf.Module):
    def __init__(self):
        super().__init__()
        # Example: simple dense layer on flattened input
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.out = tf.keras.layers.Dense(1)  # regression output

    def __call__(self, x):
        """
        x: input tensor shape [batch, 80, 8, 8], int8 dtype
        """
        x = tf.cast(x, tf.float32)  # convert to float32 for model
        x = self.flatten(x)
        x = self.dense1(x)
        return self.out(x)


def prepare_for_training(dataset, batch_size=BATCH_SIZE):
    def map_fn(tensor, p1_elo, p2_elo, p1_color):
        label = tf.stack([p1_elo, p2_elo])  # shape (2,)
        return tensor, label

    return (
        dataset
        .map(map_fn)
        .shuffle(SHUFFLE_BUFFER)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def main(tfrecord_path):
    dataset = load_tfrecord_dataset(tfrecord_path)

    # Split
    print(dataset)
    train_ds, val_ds, test_ds = split_dataset(dataset, dataset_size=42)
    # return
    train_ds = prepare_for_training(train_ds)
    val_ds = prepare_for_training(val_ds)
    test_ds = prepare_for_training(test_ds)

    model = Model()

    # Print example batch shapes
    for x_batch, y_batch in train_ds.take(1):
        print(f"Input batch shape: {x_batch.shape}")  # (batch, 80, 8, 8)
        print(f"Label batch shape: {y_batch.shape}")

    # Here you would implement training loop, optimizer, loss, etc.
    print("Ready to add training loop.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input TFRecord file")
    args = parser.parse_args()

    main(args.input)