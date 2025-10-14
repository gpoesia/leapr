#!/usr/bin/env python3

"""
Implementation of the Chess Domain abstraction.
"""

from typing import Any, Optional
import math
import random
import heapq

import chess
import numpy as np
from trainer.random_forest import RandomForestTrainer

from . import Domain
from chess_position import ChessPosition, load_chess_data
from feature_engine import Feature
from prompt_builder import load_prompt_template, format_chess_api_description

DataPoint = ChessPosition


class Chess(Domain):
    def __init__(self):
        self._split_prompt_template = load_prompt_template("prompts/chess_split.txt")
        self._funsearch_prompt_template = load_prompt_template(
            "prompts/chess_funsearch.txt"
        )

    def domain_name(self) -> str:
        return "chess"

    def load_dataset(self, path: str, max_size: int) -> list[DataPoint]:
        return load_chess_data([path], max_positions_per_file=max_size)

    def format_split_prompt(
        self,
        n_output_features: int,
        examples: list[Any],
        split_context: Optional[str],
    ) -> str:
        api = format_chess_api_description()

        def format_board_with_evaluation(pos: ChessPosition) -> str:
            return f"Board:\n{pos.board}\nEvaluation: {pos.evaluation}\n---\n"

        boards_str = "\n".join(list(map(format_board_with_evaluation, examples)))

        return self._split_prompt_template.format(
            api_description=api,
            subtree_path=split_context,
            examples=boards_str,
            num_features=n_output_features,
        )

    def format_funsearch_prompt(
        self,
        n_output_features: int,
        existing_features_with_importances: list[tuple[Feature, float]],
    ) -> str:
        api = format_chess_api_description()

        def format_features_with_importances(f: Feature, importance: float) -> str:
            return f"Feature:\n{f.code}\nImportance: {importance:.3f}\n---\n"

        features_str = (
            "\n\n".join(
                [
                    format_features_with_importances(f, imp)
                    for f, imp in existing_features_with_importances
                ]
            )
            if existing_features_with_importances
            else "<No features yet>"
        )

        return self._funsearch_prompt_template.format(
            api_description=api,
            num_features=n_output_features,
            features=features_str,
        )

    def leaf_prediction(self, datapoints: list[DataPoint]) -> float:
        if not datapoints:
            return 0.0
        return float(np.median([dp.evaluation for dp in datapoints]))

    def leaf_error(self, datapoints: list[DataPoint]) -> float:
        n = len(datapoints)
        if n <= 1:
            return 0.0
        m = self.leaf_prediction(datapoints)
        return sum(abs(dp.evaluation - m) for dp in datapoints) / n

    def code_execution_namespace(self) -> dict[str, Any]:
        return {
            "math": math,
            "chess": chess,
            "ChessPosition": ChessPosition,
            "random": random,
            "np": np,
            "numpy": np,
        }

    def best_split_for_feature(
        self,
        examples: list[DataPoint],
        feature: Feature,
        min_side_ratio: float,
    ) -> tuple[Optional[Feature], float, list[DataPoint], list[DataPoint], float]:
        return _best_split_for_feature(examples, feature, min_side_ratio)

    def input_of(self, dp: DataPoint) -> Any:
        return dp.board

    def label_of(self, dp: DataPoint) -> float:
        return dp.evaluation

    def prediction_error(self, pred: Any, label: Any) -> float:
        return abs(float(pred) - float(label))

    def train_and_evaluate_simple_predictor(
        self,
        all_features: list[Feature],
        training_set: list[DataPoint],
        validation_set: list[DataPoint],
        training_parameters: dict[str, Any] = {},
    ) -> tuple[Any, float, float]:

        trainer = RandomForestTrainer(
            features_spec={"features": [f.code for f in all_features]},
            task_type="regression",
            domain_name=self.domain_name(),
            model_type="base_predictor",
            **training_parameters,
        )

        model, metrics = trainer.train(training_set, validation_set, None)
        return model, metrics["train"]["mae"], metrics["valid"]["mae"]


class RunningMedianAbs:
    """
    Maintain a running multiset (via two heaps) to query:
      - current median
      - sum of absolute deviations from the median (in O(1))
    Push is O(log n).
    """

    def __init__(self):
        # max-heap for lower half (store negatives), min-heap for upper half
        self.low = []  # values as negative
        self.high = []  # values as positive
        self.sum_low = 0.0
        self.sum_high = 0.0

    def __len__(self):
        return len(self.low) + len(self.high)

    def push(self, x: float):
        if not self.low or x <= -self.low[0]:
            heapq.heappush(self.low, -x)
            self.sum_low += x
        else:
            heapq.heappush(self.high, x)
            self.sum_high += x
        self._rebalance()

    def _rebalance(self):
        # Keep size difference <= 1, prefer low to have the extra when odd
        if len(self.low) > len(self.high) + 1:
            x = -heapq.heappop(self.low)
            self.sum_low -= x
            heapq.heappush(self.high, x)
            self.sum_high += x
        elif len(self.high) > len(self.low):
            x = heapq.heappop(self.high)
            self.sum_high -= x
            heapq.heappush(self.low, -x)
            self.sum_low += x

    def median(self) -> float:
        return -self.low[0]

    def sum_abs_dev_from_median(self) -> float:
        """
        Let m be the median we expose (top of low).
        Sum |x - m| = m*len(low) - sum(low)  +  sum(high) - m*len(high)
        Note: low stores negatives; we keep sums in positive sense.
        """
        if len(self) == 0:
            return 0.0
        m = -self.low[0]
        left = m * len(self.low) - self.sum_low
        right = self.sum_high - m * len(self.high)
        return left + right


def _best_split_for_feature(
    examples: list[ChessPosition],
    feature: Feature,
    min_side_ratio: float,
) -> tuple[Optional[Feature], float, list[ChessPosition], list[ChessPosition], float]:
    """
    Compute the best split for a single feature.
    Returns: (feature or None, threshold, left_positions, right_positions, total_error)
    total_error = (sum_abs_dev_left + sum_abs_dev_right) / n
    """
    try:
        rows = []
        for pos in examples:
            vals = feature.execute(pos.board)
            v = vals[0] if isinstance(vals, list) else vals
            rows.append((float(v), float(pos.evaluation), pos))
    except Exception:
        # Feature execution fails, skip it
        return None, math.inf, [], [], math.inf

    if not rows:
        return None, math.inf, [], [], math.inf

    # Sort by feature value (ascending)
    rows.sort(key=lambda t: t[0])
    feats = [t[0] for t in rows]
    evals = [t[1] for t in rows]
    positions = [t[2] for t in rows]
    n = len(rows)
    if n <= 1:
        return None, math.inf, [], [], math.inf

    # Identify candidate split indices at the end of each block of equal feature values
    candidates = []
    for i in range(n - 1):  # last index where left isn't empty and right isn't empty
        if feats[i] != feats[i + 1]:
            candidates.append(i)

    if not candidates:
        # All feature values equal -> no valid split
        return None, math.inf, [], [], math.inf

    # Precompute prefix (left) sum of absolute deviations via a running median
    left_runner = RunningMedianAbs()
    left_sum_abs = [0.0] * n
    for i in range(n):
        left_runner.push(evals[i])
        left_sum_abs[i] = left_runner.sum_abs_dev_from_median()

    # Precompute suffix (right) sum of absolute deviations via a running median
    right_runner = RunningMedianAbs()
    right_sum_abs_start = [0.0] * (n + 1)  # at index j, for suffix [j..n-1]
    # right_sum_abs_start[n] = 0 (empty suffix)
    for j in range(n - 1, -1, -1):
        right_runner.push(evals[j])
        right_sum_abs_start[j] = right_runner.sum_abs_dev_from_median()

    best_error = math.inf
    best_idx = -1

    for i in candidates:
        n_left = i + 1
        n_right = n - n_left
        # min-side constraint
        if min(n_left, n_right) < min_side_ratio * n:
            continue

        sum_abs_left = left_sum_abs[i]  # for [0..i]
        sum_abs_right = right_sum_abs_start[i + 1]  # for [i+1..n-1]

        # Weighted average of mean absolute deviations equals (sum_abs_left + sum_abs_right)/n
        total_error = (sum_abs_left + sum_abs_right) / n

        if total_error < best_error:
            best_error = total_error
            best_idx = i

    if best_idx == -1:
        return None, math.inf, [], [], math.inf

    threshold = feats[best_idx]
    left_positions = positions[: best_idx + 1]
    right_positions = positions[best_idx + 1 :]

    return feature, threshold, left_positions, right_positions, best_error
