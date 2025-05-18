import chess.pgn
import json
import argparse
from tqdm import tqdm

def convert_pgn_to_json(input_path, output_path):
    games_data = []

    with open(input_path, "r") as pgn_file:
        # First count how many games there are (to initialize tqdm)
        print("Counting games...")
        game_count = 0
        while chess.pgn.read_game(pgn_file):
            game_count += 1

        pgn_file.seek(0)  # reset file pointer
        print(f"Found {game_count} games.")

        # Progress bar loop
        for _ in tqdm(range(game_count), desc="Processing PGN"):
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            # Get move list
            moves = []
            node = game
            while node.variations:
                node = node.variations[0]
                moves.append(node.san())

            # Extract headers
            white_elo = game.headers.get("WhiteElo", "Unknown")
            black_elo = game.headers.get("BlackElo", "Unknown")

            game_info = {
                "game": " ".join(moves),
                "p1_elo": white_elo,
                "p2_elo": black_elo,
                "p1_color": "white"
            }

            games_data.append(game_info)

    # Write to JSON
    with open(output_path, "w") as json_file:
        json.dump(games_data, json_file, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PGN to JSON.")
    parser.add_argument("--input", required=True, help="Path to PGN file")
    parser.add_argument("--output", required=True, help="Path to save JSON output")
    args = parser.parse_args()

    convert_pgn_to_json(args.input, args.output)