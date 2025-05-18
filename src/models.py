import tensorflow as tf

class Model_RNN(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.reshape = tf.keras.layers.Reshape((80, 64))  # reshape from (80, 8, 8) to (80, 64)

        self.bilstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, return_sequences=False)
        )  # outputs (batch_size, 128)

        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.3)

        self.out = tf.keras.layers.Dense(2)  # [p1_elo, p2_elo]

    @tf.function
    def __call__(self, x):
        x = tf.cast(x, tf.float32)  # ensure correct dtype
        x = self.reshape(x)  # (batch, 80, 64)
        x = self.bilstm(x)   # (batch, 128)
        x = self.dense1(x)
        x = self.dropout(x)
        return self.out(x)   # (batch, 2)
    


class Model_base(tf.Module):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()

        self.dense1 = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_normal')
        self.dropout1 = tf.keras.layers.Dropout(0.3)

        self.dense2 = tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_normal')
        self.dropout2 = tf.keras.layers.Dropout(0.3)

        self.dense3 = tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal')
        self.dropout3 = tf.keras.layers.Dropout(0.3)

        self.dense4 = tf.keras.layers.Dense(64, activation='relu', kernel_initializer='he_normal')

        self.out = tf.keras.layers.Dense(2)  # Output shape: (batch_size, 2)

    def __call__(self, x, training=False):
        x = tf.cast(x, tf.float32)
        x = self.flatten(x)

        x = self.dense1(x)
        x = self.dropout1(x, training=training)

        x = self.dense2(x)
        x = self.dropout2(x, training=training)

        x = self.dense3(x)
        x = self.dropout3(x, training=training)

        x = self.dense4(x)

        return self.out(x)
