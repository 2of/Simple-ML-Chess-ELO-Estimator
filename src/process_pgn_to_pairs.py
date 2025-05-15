import chess.pgn
import json

games_data = []

with open("sample_data/sample_pgn.pgn", "r") as pgn_file:
    game = chess.pgn.read_game(pgn_file)

    while game is not None:
        # Get move list
        moves = []
        node = game
        while node.variations:
            node = node.variations[0]
            moves.append(node.san())

        # Extract headers
        white_player = game.headers.get("White", "Unknown")
        black_player = game.headers.get("Black", "Unknown")
        white_elo = game.headers.get("WhiteElo", "Unknown")
        black_elo = game.headers.get("BlackElo", "Unknown")

        game_info = {
            "game": " ".join(moves),
            "p1_elo": white_elo,
            "p2_elo": black_elo,
            "p1_color": "white"  # White always plays first in chess
        }

        games_data.append(game_info)
        game = chess.pgn.read_game(pgn_file)

# Write to JSON
with open("sample_data/sample_pgn.json", "w") as json_file:
    json.dump(games_data, json_file, indent=2)