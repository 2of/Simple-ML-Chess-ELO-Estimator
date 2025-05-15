import tensorflow as tf
import numpy as np
from verify_tfrecord import load_tfrecord_dataset  # replace with your actual module

# Constants
BATCH_SIZE = 32
SHUFFLE_BUFFER = 1000
MAX_LEN = 80  # tensor depth
EPOCHS = 10
LEARNING_RATE = 1e-3

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
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.out = tf.keras.layers.Dense(2)  # Output shape (2,) for [p1_elo, p2_elo]

    def __call__(self, x):
        x = tf.cast(x, tf.float32)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.out(x)

def prepare_for_training(dataset, batch_size=BATCH_SIZE):
    def map_fn(tensor, p1_elo, p2_elo, p1_color):
        label = tf.stack([p1_elo, p2_elo])
        return tensor, label

    return (
        dataset
        .map(map_fn)
        .shuffle(SHUFFLE_BUFFER)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

def train(model, train_ds, val_ds, epochs=EPOCHS):
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    loss_fn = tf.keras.losses.MeanSquaredError()

    for epoch in range(1, epochs + 1):
        train_loss = tf.keras.metrics.Mean()
        val_loss = tf.keras.metrics.Mean()

        # Training loop
        for x_batch, y_batch in train_ds:
            with tf.GradientTape() as tape:
                preds = model(x_batch)
                loss = loss_fn(y_batch, preds)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            train_loss.update_state(loss)

        # Validation loop
        for x_batch, y_batch in val_ds:
            preds = model(x_batch)
            loss = loss_fn(y_batch, preds)
            val_loss.update_state(loss)

        print(f"Epoch {epoch}/{epochs} — Train Loss: {train_loss.result():.4f}, Val Loss: {val_loss.result():.4f}")

def main(tfrecord_path):
    dataset = load_tfrecord_dataset(tfrecord_path)

    train_ds, val_ds, test_ds = split_dataset(dataset, dataset_size=42)

    train_ds = prepare_for_training(train_ds)
    val_ds = prepare_for_training(val_ds)
    test_ds = prepare_for_training(test_ds)

    model = Model()

    # Print example batch shapes
    for x_batch, y_batch in train_ds.take(1):
        print(f"Input batch shape: {x_batch.shape}")  # (batch, 80, 8, 8)
        print(f"Label batch shape: {y_batch.shape}")  # (batch, 2)

    print("Starting training loop...")
    train(model, train_ds, val_ds, epochs=EPOCHS)

    print("Training complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input TFRecord file")
    args = parser.parse_args()

    main(args.input)