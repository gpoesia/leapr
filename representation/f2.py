#!/usr/bin/env python3

"""
Domain‑agnostic FunSearch-based representation learner (F2 in the paper, Features FunSearch)
"""

import random
import json
import logging
import warnings
import datetime
from typing import Any, Optional

import numpy as np
import wandb

from domain import Domain
from feature_engine import Feature
from llm_generator import generate_features

logger = logging.getLogger(__name__)


class FunSearchLearner:
    def __init__(
        self,
        model: str,
        n_example_top_features: int,
        n_example_random_features: int,
        features_per_iteration: int,
        num_iterations: int = 10,
        feature_testset_size: int = 10000,
        n_estimators: int = 100,
        max_depth: int = 50,
    ):
        self._model = model
        self._n_examples_top_features = n_example_top_features
        self._n_examples_random_features = n_example_random_features
        self._features_per_iteration = features_per_iteration
        self._num_iterations = num_iterations
        self._feature_testset_size = feature_testset_size

        self._trainer_kwargs = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
        )

    def learn_features(
        self,
        domain: Domain,
        training_set: list[Any],
        validation_set: list[Any],
    ) -> list[str]:
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        ckpt_name = f"funsearch_checkpoint_{ts}-{domain.domain_name()}.json"
        logger.info(f"Checkpoint file: {ckpt_name}")

        all_features: list[Feature] = []
        importances = []
        prev_valid_error: Optional[float] = None

        def _after_retrain_log_and_ckpt(valid_error: float):
            nonlocal prev_valid_error
            err_delta = (valid_error - prev_valid_error) if prev_valid_error is not None else 0.0
            wandb.log({"valid_error": valid_error, "error_delta": err_delta})
            logger.info(
                f"Error on validation set after retrain: {valid_error:.4f}"
                + (f" (delta {err_delta:+.4f})" if prev_valid_error is not None else "")
            )
            prev_valid_error = valid_error

            try:
                with open(ckpt_name, "w") as f:
                    json.dump({ "all_features": [f_.code for f_ in all_features] }, f)
            except Exception as e:
                logger.warning(f"Failed to write checkpoint {ckpt_name}: {e}")

        for it in range(self._num_iterations):
            logger.info(f"Iteration {it+1}/{self._num_iterations}")
            logger.info("=" * 60)

            # 1) Propose features using LLM
            # 1a. Choose feature examples to show in the prompt.
            example_features_with_importances = []
            sorted_features = sorted(
                zip(all_features, importances),
                key=lambda fi: fi[1],
                reverse=True,
            )
            example_features_with_importances.extend(
                sorted_features[: self._n_examples_top_features]
            )
            example_features_with_importances.extend(
                random.sample(
                    sorted_features[self._n_examples_top_features :],
                    max(0, min(self._n_examples_random_features,
                               len(sorted_features) - self._n_examples_top_features))
                )
            )

            random.shuffle(example_features_with_importances)

            prompt = domain.format_funsearch_prompt(
                n_output_features=self._features_per_iteration,
                existing_features_with_importances=example_features_with_importances,
            )

            # 1b. Get LLM proposals.
            feature_codes: list[str] = generate_features(
                model=self._model,
                prompt=prompt,
            )

            # 2) Validate feature execution on a sample of positions
            test_slice = training_set[: self._feature_testset_size]
            new_features: list[Feature] = []
            for code in feature_codes:
                try:
                    warnings.simplefilter('error')
                    f = Feature(code, domain)
                    for dp in test_slice:
                        assert np.isfinite(f.execute(domain.input_of(dp))).all(), \
                            "Feature produced non-finite values"
                    new_features.append(f)
                except Exception as e:
                    logger.info(f"Skipping feature (execution failed): {e}\n{code}")

            if new_features:
                logger.info("New features:")
                for f in new_features:
                    logger.info(f" • {f.code}")
            else:
                logger.info("No working features generated this round.")

            # 3) Add all features to the model and retrain
            all_features.extend(new_features)
            _model, _, valid_e = domain.train_and_evaluate_simple_predictor(
                all_features,
                training_set,
                validation_set,
                self._trainer_kwargs,
            )
            importances = _model.feature_importances_.tolist()
            _after_retrain_log_and_ckpt(valid_e)


        return [f.code for f in all_features]
