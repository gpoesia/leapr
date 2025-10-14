#!/usr/bin/env python3

import logging
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from .trainer import Trainer
from .utils import (
    evaluate_regression_model,
    evaluate_classification_model,
    prepare_train_valid_split,
)

from feature_engine import prepare_supervised_data
from value import RFValueFunction
from typing import Any, Optional
import json


logger = logging.getLogger(__name__)


class RandomForestTrainer(Trainer):
    """Random Forest trainer implementation for regression and classification."""

    def __init__(
        self,
        features_spec: Any,
        task_type: str = "regression",
        domain_name: str = "chess",
        n_estimators: int = 100,
        random_state: int = 42,
        max_depth: int = 5,
        min_samples_leaf: int = 5,
        model_type: str = "value_function",
        **kwargs,
    ):

        if "features" in features_spec:
            self.features = features_spec["features"]
        else:
            features_file, features_key = features_spec["file"], features_spec.get(
                "key", "used_features"
            )
            with open(features_file, "r") as f:
                data = json.load(f)
                if features_key is not None:  # key not null
                    self.features = data[features_key]
                elif "used_features" in data:
                    self.features = data["used_features"]  # default
                else:
                    raise ValueError(f"No valid key found in {features_file}")

        self.task_type = task_type
        self.domain_name = domain_name
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.model_type = model_type

    def train(
        self,
        train_positions: list,
        valid_positions: list,
        eval_positions: Optional[list] = None,
    ):
        """Train a Random Forest and return (model/value_function, metrics)."""
        logger.info(
            f"Training Random Forest for {self.task_type} with {len(self.features)} features"
        )

        X_train, y_train, X_valid, y_valid = prepare_train_valid_split(
            self.features, train_positions, valid_positions, self.domain_name
        )

        if self.task_type == "classification":
            model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
                n_jobs=-1,
            )
            evaluate_fn = evaluate_classification_model
            metric_key = "accuracy"
        else:
            model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
                n_jobs=-1,
            )
            evaluate_fn = evaluate_regression_model
            metric_key = "mae"

        logger.info(f"Training RF for {self.task_type}...")
        model.fit(X_train, y_train)
        logger.info("Done. Evaluating...")

        train_metrics = evaluate_fn(model, X_train, y_train)
        valid_metrics = evaluate_fn(model, X_valid, y_valid)

        eval_metrics = None
        if eval_positions is not None:
            X_test, y_test = prepare_supervised_data(
                self.features, eval_positions, self.domain_name
            )
            eval_metrics = evaluate_fn(model, X_test, y_test)

        logger.info("Done evaluating.")

        logger.info(f"📊 Training Performance:")
        logger.info(f"   Training {metric_key}: {train_metrics[metric_key]:.4f}")
        logger.info(f"   Validation {metric_key}: {valid_metrics[metric_key]:.4f}")

        if eval_metrics:
            logger.info(f"   Evaluation {metric_key}: {eval_metrics[metric_key]:.4f}")

        all_metrics = {
            "train": train_metrics,
            "valid": valid_metrics,
            "eval": eval_metrics,
        }

        if self.model_type == "base_predictor":
            return model, all_metrics

        assert (
            self.model_type == "value_function"
        ), f"Unknown model_type: {self.model_type}"
        value_function = RFValueFunction(model, self.features)
        return value_function, all_metrics
