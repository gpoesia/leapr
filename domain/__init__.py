#!/usr/bin/env python3

"""
Wraps all domain-specific parts of running funsearch and D-ID3.
"""

from abc import ABC
from typing import Any, Optional
from feature_engine import Feature

DataPoint = Any

class Domain(ABC):
    def domain_name(self) -> str:
        raise NotImplementedError

    def load_dataset(self, path: str, max_size: int) -> list[DataPoint]:
        raise NotImplementedError

    def format_funsearch_prompt(
        self,
        n_output_features: int,
        existing_features_with_importances: list[tuple[Feature, float]],
    ) -> str:
        raise NotImplementedError


    def format_split_prompt(
        self,
        n_output_features: int,
        examples: list[Any],
        split_context: Optional[str],
    ) -> str:
        raise NotImplementedError

    def leaf_error(
        self,
        datapoints: list[DataPoint]
    ) -> float:
        raise NotImplementedError

    def leaf_prediction(
        self,
        datapoints: list[DataPoint]
    ) -> DataPoint:
        raise NotImplementedError

    def code_execution_namespace(self) -> dict[str, DataPoint]:
        raise NotImplementedError

    def best_split_for_feature(
        self,
        examples: list[DataPoint],
        feature: Feature,
        min_side_ratio: float,
    ) -> tuple[Optional[Feature], float, list[DataPoint], list[DataPoint], float]:
        raise NotImplementedError

    def input_of(self, dp: DataPoint) -> Any:
        raise NotImplementedError

    def label_of(self, dp: DataPoint) -> float:
        raise NotImplementedError

    def prediction_error(self, pred: Any, label: Any) -> float:
        raise NotImplementedError

    def train_and_evaluate_simple_predictor(
        self,
        all_features: list[Feature],
        training_set: list[DataPoint],
        validation_set: list[DataPoint],
        training_parameters: dict[str, Any] = {}
    ) -> tuple[Any, float, float]:
        raise NotImplementedError
