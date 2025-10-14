#!/usr/bin/env python3

from abc import ABC, abstractmethod
from typing import Any, Optional

class Trainer(ABC):
    """Abstract base class for training models."""

    @abstractmethod
    def train(
        self,
        train_positions: list,
        valid_positions: list,
        eval_positions: Optional[list] = None,
    ) -> tuple[Any, dict]:
        """Train a value function and return (value_function, metrics)."""
        pass
