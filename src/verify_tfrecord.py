import tensorflow as tf

def _parse_function(example_proto):
    feature_description = {
        'tensor': tf.io.FixedLenFeature([], tf.string),
        'p1_elo': tf.io.FixedLenFeature([], tf.float32),
        'p2_elo': tf.io.FixedLenFeature([], tf.float32),
        'p1_color': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)

    tensor = tf.io.parse_tensor(parsed['tensor'], out_type=tf.int8)
    tensor = tf.reshape(tensor, [80, 8, 8])

    p1_elo = parsed['p1_elo']
    p2_elo = parsed['p2_elo']
    p1_color = parsed['p1_color']  # 1 = white, 0 = black

    return tensor, p1_elo, p2_elo, p1_color


def load_tfrecord_dataset(tfrecord_path):

    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    parsed_dataset = raw_dataset.map(_parse_function)
    return parsed_dataset


def print_dataset_summary(dataset, max_samples=10):

    for i, (tensor, p1_elo, p2_elo, p1_color) in enumerate(dataset):
        print(f"Game {i + 1}:")
        print(f"  Tensor shape: {tensor.shape}")
        print(f"  p1_elo: {p1_elo.numpy()}")
        print(f"  p2_elo: {p2_elo.numpy()}")
        print(f"  p1_color: {'white' if p1_color.numpy() == 1 else 'black'}")
        print(f"  Tensor sample (first move):\n{tensor[0].numpy()}")
        print("-" * 40)
        if i + 1 >= max_samples:
            break


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input TFRecord file")
    args = parser.parse_args()

    dataset = load_tfrecord_dataset(args.input)
    print_dataset_summary(dataset)