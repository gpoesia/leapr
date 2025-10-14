#!/usr/bin/env python3

import os
import json
import random
import math
import time
from concurrent.futures import ProcessPoolExecutor

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from stockfish import Stockfish
from tqdm import tqdm
from chess import Board

from policy import ValueSoftmax, from_config, Policy
from chess_position import load_chess_data
from sequential_executor import SequentialExecutor
from trainer.utils import evaluate_regression_model


class PolicyEvaluator:
    def evaluate(self, policy: Policy) -> tuple[float, dict]:
        """Evaluate the policy. Returns (score, stats dict)."""
        raise NotImplementedError

cache_dirty = False


def eval_pos(self, pos, policy, stockfish_predictions):
    fen = pos.fen
    # assert pos.board.legal_moves.count() > 1

    top_moves = sorted(policy(pos.board).items(), key=lambda kv: kv[1], reverse=True)

    if fen not in stockfish_predictions:
        print('Running stockfish...')
        all_top_moves = []

        for _ in range(self._stockfish_runs):
            sf = self._get_stockfish()
            sf.set_fen_position(fen)
            all_top_moves.append(sf.get_best_move_time(self._stockfish_time))

        stockfish_predictions[fen] = all_top_moves
        global cache_dirty
        cache_dirty = True

    policy_top_move = top_moves[0][0]

    if policy_top_move in stockfish_predictions[fen]:
        return True

    return False


class AccuracyEvaluator(PolicyEvaluator):
    def __init__(self,
                 eval_positions: list[str],
                 top_k: int,
                 n_positions: int,
                 stockfish_top_moves_cache: str,
                 output: str,
                 stockfish_time: int = 500,
                 stockfish_runs: int = 10,
                 n_jobs=0,
                 ):
        engine_path = os.path.join(os.path.dirname(__file__), "stockfish", "stockfish")
        self.engine_path = engine_path
        self._stockfish_time = stockfish_time
        self._top_k = top_k
        self._n_positions = n_positions
        self._cache_path = stockfish_top_moves_cache
        self._n_jobs = n_jobs
        self._output = output
        # Load games from eval_positions file. Divide moves by 10 to get an approximate number of games we need.
        self._positions = load_chess_data(eval_positions, self._n_positions)

        random.seed('accuracy-eval')
        random.shuffle(self._positions)

        self._stockfish_runs = stockfish_runs


    def _get_stockfish(self):
        return Stockfish(path=self.engine_path,
                depth=25,
                parameters={
                    'Threads': 1,
                    'Hash': 1024,
                    'UCI_Elo': 3190,  # Max value according to documentation.
                    })

    def evaluate(self, policy: Policy) -> tuple[float, dict]:
        stockfish_predictions = {}
        global cache_dirty

        if os.path.exists(self._cache_path):
            with open(self._cache_path, "r") as f:
                stockfish_predictions = json.load(f)

        executor = (SequentialExecutor()
                    if self._n_jobs == 1
                    else ProcessPoolExecutor(max_workers=self._n_jobs or os.cpu_count()))

        with executor as ex:
            matches = list(
                        ex.map(
                            eval_pos,
                            tqdm([self] * len(self._positions), desc='Computing accuracy against Stockfish...'), 
                            self._positions, 
                            [policy] * len(self._positions), 
                            [stockfish_predictions] * len(self._positions)),
                    )

        if cache_dirty:
            with open(self._cache_path, "w") as f:
                json.dump(stockfish_predictions, f, indent=2)

        total = len(self._positions)
        accuracy = sum(matches) / total if total else float("nan")
        stats = {'matches': sum(matches), 'total': total}

        print(f"Policy matches Stockfish top move in {stats['matches']}/{total} positions ({accuracy:.2%})")

        if self._output:
            with open(self._output, 'w') as f:
                json.dump({'accuracy': accuracy, **stats}, f)
            print('Wrote', self._output)

        return accuracy, stats


class BoardEvaluationEvaluator(PolicyEvaluator):
    def __init__(self,
                 eval_positions: list[str],
                 n_positions: int,
                 output: str):
        self._n_positions = n_positions
        self._positions = load_chess_data(eval_positions, self._n_positions // 10)
        self._output = output

        random.seed('eval-eval')
        random.shuffle(self._positions)

    def evaluate(self, policy: Policy) -> tuple[float, dict]:
        assert isinstance(policy, ValueSoftmax), 'Policy must be ValueSoftmax for BoardEvaluationEvaluator'

        y_true = np.array([pos.evaluation for pos in self._positions])
        X = [pos.board for pos in self._positions]
        stats = evaluate_regression_model(policy._value_function, X, y_true)

        print('RMSE:', stats['rmse'])
        print('Rho:', stats['rho'])

        if self._output:
            with open(self._output, "w") as f:
                json.dump(stats, f, indent=2)
            print(f"Saved evaluation stats to {self._output}")

        return stats['rmse'], stats


@hydra.main(version_base=None, config_path="config", config_name="evaluation")
def main(cfg: DictConfig) -> None:
    policy = from_config(cfg.policy)

    evaluator = hydra.utils.instantiate(cfg.evaluator)
    evaluator.evaluate(policy)

if __name__ == "__main__":
    main()
