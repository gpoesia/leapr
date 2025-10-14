#!/usr/bin/env python3

import math
import os
import pickle
from typing import Callable, Optional, Any
import time

import torch
import numpy as np
from omegaconf import DictConfig
from chess import Board, BLACK
from stockfish import Stockfish

from feature_engine import execute_feature
from chess_position import ChessPosition
from value import RFValueFunction

from domain.game_state import GameState
from domain.chess_game_state import ChessGameState


class Policy:
    """
    A Policy takes a state and returns a probability distribution over actions.
    """

    def __call__(self, _: Board) -> dict[str, float]:
        """
        Given a state, return a probability distribution over actions.
        :param state: The current state.
        :return: A dictionary mapping actions (moves in UCI notation) to their probabilities.
        """
        raise NotImplementedError


class Uniform(Policy):
    """
    A uniform policy that returns equal probability for all actions.
    """

    def __call__(self, state: Board) -> dict[str, float]:
        moves = state.legal_moves
        p = 1 / moves.count() if moves else 0
        return {m.uci(): p for m in moves}


class ValueSoftmax(Policy):
    """
    A policy derived from a value function by softmax over values of immediate successor states.
    """

    def __init__(
        self, value_function: Callable[[list[Board]], list[float]], tau: float = 50
    ):
        """
        Initialize with a value function.
        :param value_function: A callable that returns values for a list of board positions.
        """
        self._value_function = value_function
        self._tau = tau

    def __call__(self, state: Board) -> dict[str, float]:
        moves = list(state.legal_moves)
        if not moves:
            return {}

        successors = [state.copy() for _ in moves]
        for m, s in zip(moves, successors):
            s.push(m)

        # The values computed for the successors are the values for the opponent.
        # Thus, we negate to get their values for the current player.
        values = -np.array(self._value_function(successors)) / self._tau

        # Compute softmax with standard numerical stability trick.
        exp_values = np.exp(values - np.max(values) + 1e-4)
        pi = exp_values / np.sum(exp_values)

        return {m.uci(): p for m, p in zip(moves, pi)}


class StockfishWithELO(Policy):
    """
    Policy that simply calls Stockfish with a given setup (e.g., max depth, estimated ELO, etc).
    """

    def __init__(self, elo: int, tau: float = 0.2):
        self.engine_path = os.path.join(os.path.dirname(__file__), "stockfish", "stockfish")
        self.elo = elo
        self._tau = tau

    def __call__(self, state: Board) -> dict[str, float]:
        moves = list(state.legal_moves)
        if not moves:
            return {}

        sf = Stockfish(path=self.engine_path, depth=25, parameters = {
            'Threads': 1,
            'Hash': 1024,
            'UCI_Elo': self.elo,
        })

        sf.set_fen_position(state.fen())
        top_move = sf.get_best_move_time(500)

        # Compute softmax with standard numerical stability trick.
        values = np.exp([1 / self._tau if m.uci() == top_move else 0.0 for m in moves])
        pi = values / values.sum()

        return {m.uci(): p for m, p in zip(moves, pi)}


class MCTSNode:
    def __init__(
        self,
        state: GameState,
        parent: Optional["MCTSNode"] = None,
        move: Optional[Any] = None,
    ):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = {}
        self.visits = 0
        self.value_sum = 0.0
        self.expanded = False

    def is_fully_expanded(self) -> bool:
        return self.expanded and len(self.children) == len(self.state.get_legal_moves())

    def best_child(
        self, c_param: float = 1.414, selection_strategy: str = "ucb1"
    ) -> "MCTSNode":
        """Select best child using specified strategy."""
        choices_weights = []
        for child in self.children.values():
            if child.visits == 0:
                return child

            if selection_strategy == "ucb1":
                weight = (child.value_sum / child.visits) + c_param * math.sqrt(
                    2 * math.log(self.visits) / child.visits
                )
            elif selection_strategy == "ucb1_tuned":
                mean = child.value_sum / child.visits
                variance = (child.value_sum**2 / child.visits) - mean**2
                v = variance + math.sqrt(2 * math.log(self.visits) / child.visits)
                weight = mean + math.sqrt(
                    math.log(self.visits) / child.visits * min(0.25, v)
                )
            else:
                raise ValueError(f"Unknown selection strategy: {selection_strategy}")

            choices_weights.append(weight)

        return list(self.children.values())[np.argmax(choices_weights)]

    def expand(self) -> "MCTSNode":
        """Expand by adding one child."""
        if self.state.is_terminal():
            self.expanded = True
            return self

        legal_moves = self.state.get_legal_moves()
        for move in legal_moves:
            if move not in self.children:
                child_state = self.state.make_move(move)
                child = MCTSNode(child_state, parent=self, move=move)
                self.children[move] = child
                if len(self.children) == len(legal_moves):
                    self.expanded = True
                return child

        self.expanded = True
        return self

    def backpropagate(self, value: float):
        """Backpropagate value through the tree."""
        self.visits += 1
        self.value_sum += value
        if self.parent:
            # Negate value for opponent (parent)
            self.parent.backpropagate(-value)


class MCTS(Policy):
    """Monte Carlo Tree Search policy."""

    def __init__(
        self,
        value_function: Callable[[list[Board]], list[float]],
        simulations: int = 1000,
        c_param: float = 1.414,
        selection_strategy: str = "ucb1",
        evaluation_callback: Callable = None,
        evaluation_interval: int = None,
    ):
        """
        Initialize MCTS policy.
        :param value_function: Function that evaluates positions
        :param simulations: Number of MCTS simulations to run
        :param c_param: UCB1 exploration parameter
        :param selection_strategy: Selection strategy ("ucb1", "ucb1_tuned")
        :param evaluation_callback: Function to evaluate accuracy during search
        :param evaluation_interval: How often to run the evaluation
        """
        self.value_function = value_function
        self.simulations = simulations
        self.c_param = c_param
        self.selection_strategy = selection_strategy
        self.evaluation_callback = evaluation_callback

        if evaluation_interval is None:
            self.evaluation_interval = max(
                1, self.simulations // 10
            )  # Default to simulations/10
        else:
            self.evaluation_interval = evaluation_interval

    def __call__(self, state: Board) -> dict[str, float]:
        # Convert chess.Board to GameState for generality
        game_state = ChessGameState(state)
        return self._search(game_state)

    def _search(self, game_state: GameState) -> dict[str, float]:
        """Generic MCTS search that works with any GameState."""
        if game_state.is_terminal():
            return {}

        root = MCTSNode(game_state)

        # Run MCTS simulations
        for sim in range(self.simulations):
            node = self._select(root)
            if not node.state.is_terminal() and not node.is_fully_expanded():
                node = node.expand()
            value = self._evaluate(node)
            node.backpropagate(value)

            # Calls the callback every evaluation_interval with current simulation count and root node
            if self.evaluation_callback and (sim + 1) % self.evaluation_interval == 0:
                self.evaluation_callback(sim + 1, root)

        # Convert visit counts to probabilities
        total_visits = sum(child.visits for child in root.children.values())
        if total_visits == 0:
            # Fallback to uniform distribution
            moves = game_state.get_legal_moves()
            uniform_prob = 1.0 / len(moves) if moves else 0.0
            return {game_state.move_to_string(move): uniform_prob for move in moves}

        return {
            game_state.move_to_string(move): child.visits / total_visits
            for move, child in root.children.items()
        }

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Selection phase: traverse tree using specified strategy."""
        while not node.state.is_terminal():
            if not node.is_fully_expanded():
                return node
            node = node.best_child(self.c_param, self.selection_strategy)
        return node

    def _evaluate(self, node: MCTSNode) -> float:
        """Evaluation phase: get value estimate for the position."""
        if node.state.is_terminal():
            return node.state.get_terminal_value()

        # For chess, extract the board for the value function
        if isinstance(node.state, ChessGameState):
            values = self.value_function([node.state.board])
        else:
            # For other domains, pass the state directly
            values = self.value_function([node.state])

        # The values computed for the successors are the values for the opponent.
        # Thus, we negate to get their values for the current player.
        return -values[0] if values else 0.0


def from_config(config: DictConfig) -> Policy:
    """
    Create a Policy instance from a configuration.
    :param config: Configuration object.
    :return: The corresponding Policy.
    """
    t = config.type.lower()

    value_function = None

    if 'model_path' in config:
        if config.model_path.endswith('.pkl'):
            with open(config.model_path, "rb") as f:
                obj = pickle.load(f)
                if isinstance(obj, RFValueFunction):
                    value_function = obj
                else:
                    value_function, _training_stats = obj
        elif config.model_path.endswith('.pt'):
            obj = torch.load(config.model_path, weights_only=False)
            if isinstance(obj, dict) and 'model' in obj:
                value_function = obj['model']
                # Unwrap DataParallel - compatibility with some old checkpoints
                if isinstance(value_function.model, torch.nn.DataParallel):
                    value_function.model = value_function.model.module
            else:
                value_function, _training_stats = obj

    if t == "uniform":
        return Uniform()
    elif t == "stockfish_with_elo":
        return StockfishWithELO(elo=config.elo,
                                tau=config.get("tau", 0.2))
    elif t == "value_softmax":
        if value_function is None:
            raise ValueError("ValueSoftmax policy requires a value_function")
        tau = config.get("tau", 50)
        return ValueSoftmax(value_function, tau=tau)
    elif t == "mcts":
        if value_function is None:
            raise ValueError("MCTS policy requires a value_function")
        simulations = config.get("simulations", 1000)
        c_param = config.get("c_param", 1.414)
        selection_strategy = config.get("selection_strategy", "ucb1")
        evaluation_interval = config.get("evaluation_interval", None)
        return MCTS(
            value_function,
            simulations=simulations,
            c_param=c_param,
            selection_strategy=selection_strategy,
            evaluation_interval=evaluation_interval,
        )
    else:
        raise ValueError(f"Unknown policy type: {config.type}")
