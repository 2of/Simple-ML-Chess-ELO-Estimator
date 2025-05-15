'''
Generates an 8×8×80 tensor from the moves played.
Pads or truncates games as needed. Saves output as TFRecord.
'''

import chess
import json
import numpy as np
import tensorflow as tf

MAX_LEN = 80  # Fixed tensor depth

PIECE_MAP = {
    "P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6,
    "p": -1, "n": -2, "b": -3, "r": -4, "q": -5, "k": -6
}


def board_to_array(board):
    array = np.zeros((8, 8), dtype=np.int8)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = 7 - chess.square_rank(square)
            col = chess.square_file(square)
            array[row, col] = PIECE_MAP[piece.symbol()]
    return array


def game_to_tensor(move_string, max_len=MAX_LEN):
    board = chess.Board()
    moves = move_string.split()
    tensors = []

    for move in moves:
        try:
            board.push_san(move)
        except Exception:
            break  # skip illegal moves
        tensors.append(board_to_array(board))

    if len(tensors) >= max_len:
        tensors = tensors[:max_len]
    else:
        padding = [np.zeros((8, 8), dtype=np.int8) for _ in range(max_len - len(tensors))]
        tensors += padding

    return np.stack(tensors)  # shape: [80, 8, 8]


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(tensor, p1_elo, p2_elo, p1_color):
    feature = {
        'tensor': _bytes_feature(tf.io.serialize_tensor(tensor).numpy()),
        'p1_elo': _float_feature(p1_elo),
        'p2_elo': _float_feature(p2_elo),
        'p1_color': _int64_feature(0 if p1_color == "black" else 1),  # white=1, black=0
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def main(json_path, output_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    with tf.io.TFRecordWriter(output_path) as writer:
        for i, entry in enumerate(data):
            move_str = entry["game"]
            try:
                tensor = game_to_tensor(move_str)
                p1_elo = float(entry["p1_elo"])
                p2_elo = float(entry["p2_elo"])
                p1_color = entry["p1_color"].strip().lower()

                if p1_color not in {"white", "black"}:
                    raise ValueError(f"Invalid p1_color: {p1_color}")

                example = serialize_example(tensor, p1_elo, p2_elo, p1_color)
                writer.write(example.SerializeToString())
                print(f"Processed Game {i+1}: shape = {tensor.shape}")
            except Exception as e:
                print(f"Skipping game {i+1} due to error: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input JSON file")
    parser.add_argument("--output", required=True, help="Path to output TFRecord file")
    args = parser.parse_args()

    main(args.input, args.output)