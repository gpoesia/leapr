import json
import logging
import random
from typing import Optional
import os

import sys
import chess.pgn
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

PARSE_BOARD = not bool(os.environ.get('DONT_PARSE_BOARD', False))


def eval_to_win_probability(eval_cp: float) -> float:
    """Convert centipawn evaluation to win probability using lichess formula."""
    # Source: https://lichess.org/page/accuracy
    return 100 / (1 + np.exp(-0.00368208 * eval_cp))


class ChessPosition:
    """Represents a chess position with board state and evaluation."""

    def __init__(
        self,
        board: chess.Board,
        evaluation: float,
        move_history: Optional[list[str]] = None,
        game_phase: Optional[str] = None,  # NEW: Added optional game phase
        fen: str = None,
    ):
        self.board = board.copy() if board else None
        self.evaluation = evaluation  # Stockfish evaluation in centipawns
        self.move_history = move_history or []
        self.fen = board.fen() if PARSE_BOARD else fen
        self.game_phase = game_phase or "unknown"  # NEW: Store game phase

    def __str__(self):
        # NEW: Include phase in string representation
        phase_str = f" ({self.game_phase})" if self.game_phase != "unknown" else ""
        return f"FEN: {self.fen}\nEvaluation: {self.evaluation}\nMove History: {' '.join(self.move_history[-10:])}{phase_str}"


def load_chess_data(
    data_paths: list[str], max_positions_per_file: int = 1000
) -> list[ChessPosition]:
    """Load chess positions from the specified dataset files."""
    logger.info(f"Loading chess data from {len(data_paths)} files")
    positions = []

    for path in data_paths:
        logger.info(f"Loading from {path}")

        if path.endswith(".json"):
            positions.extend(_load_from_json(path, max_positions_per_file))
        elif path.endswith(".jsonl"):
            positions.extend(_load_from_jsonl(path, max_positions_per_file))
        elif path.endswith(".pgn"):
            positions.extend(_load_from_pgn(path, max_positions_per_file))
        else:
            logger.warning(f"Unsupported file format: {path}")

    logger.info(f"Loaded {len(positions)} chess positions")

    return positions


def _load_from_jsonl(path: str, max_positions_per_file: int) -> list[ChessPosition]:
    """Load positions from JSON file with the new format."""
    positions = []

    with open(path, "r") as f:
        for line in tqdm(f, maxinterval=max_positions_per_file, desc="Loading positions"):
            if len(positions) >= max_positions_per_file:
                break

            data = json.loads(line.strip())
            fen = data['fen']

            stockfish_evals = data['evals']
            # Lichess recommendation: take the evaluation with highest depth,
            # and take its first PV (Principal Evaluation)
            # See https://database.lichess.org/#evals
            stockfish_evals.sort(key=lambda x: x["depth"], reverse=True)
            first_pv_eval = stockfish_evals[0]['pvs'][0]
            if 'cp' in first_pv_eval:
                evaluation = float(first_pv_eval['cp'])
            elif 'mate' in first_pv_eval:
                # Large negative value indicating "bad" position
                evaluation = 1000 * float(np.sign(first_pv_eval['mate']))
            else:
                raise ValueError(f"Need either 'cp' or 'mate' in evaluation data: {first_pv_eval}")

            evaluation = max(-1000, min(1000, evaluation))  # Clamp to [-1000, 1000]

            game_phase = data.get("game_phase", "unknown")
            board = chess.Board(fen) if PARSE_BOARD else None
            position = ChessPosition(board, eval_to_win_probability(evaluation), move_history=None, game_phase=game_phase, fen=fen)
            positions.append(position)

    return positions


def _load_from_json(path: str, max_positions_per_file: int) -> list[ChessPosition]:
    """Load positions from JSON file with the new format."""
    with open(path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]

    # Sample if too many positions
    if len(data) > max_positions_per_file:
        data = random.sample(data, max_positions_per_file)

    positions = []
    for position_data in data:
        try:
            # Handle new JSON format
            fen = position_data["fen"]

            # Handle potentially None stockfish_evaluation
            stockfish_eval = position_data.get("stockfish_evaluation")
            mate_moves = position_data.get("mate_in_moves")

            if stockfish_eval is not None:
                evaluation = float(stockfish_eval) * 100  # Convert to centipawns
            elif mate_moves is not None:
                evaluation = 1000 * float(np.sign(mate_moves))
            else:
                # Skip positions without any evaluation
                continue

            # Extract move history if available
            move_history = []
            if "position_context" in position_data:
                # Could parse move history from context if needed
                pass

            # NEW: Extract game phase if available
            game_phase = position_data.get("game_phase", "unknown")

            board = chess.Board(fen) if PARSE_BOARD else None
            position = ChessPosition(board, eval_to_win_probability(evaluation), move_history, game_phase, fen)
            positions.append(position)
        except Exception as e:
            # Silently skip problematic positions
            continue

    return positions


def _load_from_pgn(path: str, max_positions_per_file: int) -> list[ChessPosition]:
    with open(path, "r") as f:
        games = []

        print("Reading games")

        while True:
            game = chess.pgn.read_game(f)
            if game is None or len(games) >= max_positions_per_file:
                break
            games.append(game)

        print("Generating board positions")
        positions = []
        for game in games:
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
                evaluation = 0.0
                position = ChessPosition(
                    board,
                    eval_to_win_probability(evaluation),
                    move_history=None,
                    game_phase=None,
                    fen=board.fen(),
                )
                positions.append(position)

        return positions
