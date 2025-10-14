#!/usr/bin/env python3

from abc import ABC, abstractmethod
from typing import Any, List


class GameState(ABC):
    """Abstract interface for generic game states."""

    @abstractmethod
    def copy(self) -> "GameState":
        """Return a copy of this state."""
        pass

    @abstractmethod
    def get_legal_moves(self) -> List[Any]:
        """Return list of legal moves from this state."""
        pass

    @abstractmethod
    def make_move(self, move: Any) -> "GameState":
        """Return new state after making a move."""
        pass

    @abstractmethod
    def is_terminal(self) -> bool:
        """Return True if this is a terminal state."""
        pass

    @abstractmethod
    def get_terminal_value(self) -> float:
        """Return value for terminal states (from current player's perspective)."""
        pass

    @abstractmethod
    def move_to_string(self, move: Any) -> str:
        """Convert move to string representation."""
        pass
