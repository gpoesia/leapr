#!/usr/bin/env python3

import chess
from typing import List

from domain.game_state import GameState


class ChessGameState(GameState):
    def __init__(self, board: chess.Board):
        self.board = board

    def copy(self) -> "ChessGameState":
        return ChessGameState(self.board.copy())

    def get_legal_moves(self) -> List[chess.Move]:
        return list(self.board.legal_moves)

    def make_move(self, move: chess.Move) -> "ChessGameState":
        new_board = self.board.copy()
        new_board.push(move)
        return ChessGameState(new_board)

    def is_terminal(self) -> bool:
        return self.board.is_game_over()

    def get_terminal_value(self) -> float:
        if not self.board.is_game_over():
            return 0.0

        result = self.board.result()
        # We want value from perspective of player who moved TO this position
        # That's the opposite of self.board.turn (current player to move)
        if result == "1-0":  # White won
            return -1000.0 if self.board.turn == chess.WHITE else 1000.0
        elif result == "0-1":  # Black won
            return 1000.0 if self.board.turn == chess.WHITE else -1000.0
        else:
            return 0.0

    def move_to_string(self, move: chess.Move) -> str:
        return move.uci()
