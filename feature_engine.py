import functools
import logging
import math
import random
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import Any
from itertools import repeat
import warnings

import numpy as np
from tqdm import tqdm

from chess_position import ChessPosition
from image_sample import ImageSample

logger = logging.getLogger(__name__)

SILENCE_FEATURE_ERRORS = False
FEATURE_ERROR_VALUE = -1e9


class Feature:
    def __init__(self, code: str, domain=None):
        self.code = code
        self.domain = domain
        self.feature = load_feature(code)
        self._cache = {}

    def execute(self, data: Any) -> list[float]:
        """
        Execute the feature function on data
        """
        data_id = id(data)
        value = self._cache.get(data_id)

        if value is None:
            value = self._cache[data_id] = execute_feature(self.code, data, self.domain)
        return value

    @property
    def description(self) -> str:
        """
        Get a description of the feature function.
        """
        return self.feature.__doc__

    def __hash__(self):
        return hash(self.code)


@functools.cache
def load_feature(feature_code: str, domain=None):
    """Execute a feature function given as a string."""
    if domain is not None:
        namespace = domain.code_execution_namespace()
    else:
        # Fallback for backward compatibility
        import chess
        from chess_position import ChessPosition

        namespace = {
            "math": math,
            "chess": chess,
            "ChessPosition": ChessPosition,
            "random": random,
            "np": np,
            "numpy": np,
        }

    exec(feature_code, namespace)
    return namespace["feature"]


def execute_feature(feature_code: str, data: Any, domain=None) -> list[float]:
    """Execute a feature function on data."""
    try:
        f = load_feature(feature_code, domain)
        v = f(data)

        if not np.isfinite(v).all():
            raise ValueError('Feature not finite.')

        return [v] if not isinstance(v, list) else v
    except Exception as e:
        if SILENCE_FEATURE_ERRORS:
            return [FEATURE_ERROR_VALUE]
        logger.debug(f"Error executing feature: {e}")  # Changed from error to debug
        raise


def _compute_row(features: list[str], data: Any, domain=None) -> list[float]:
    """Compute concatenated feature values for a single board (worker-safe)."""
    vals: list[float] = []
    for feature in features:
        vals.extend(execute_feature(feature, data, domain))
    return vals


def prepare_supervised_data(
    features: list[str],
    samples: list,
    domain_name: str = "chess",
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare training data for supervised learning (classification or regression)."""
    if not samples:
        return np.array([]), np.array([])

    logger.info("Computing features in parallel...")
    max_workers = os.cpu_count() or 1

    if domain_name == "image_classification":
        from domain.image_classification import ImageClassification

        domain = ImageClassification()
        data_points = [s.image for s in samples]
        targets = [s.target for s in samples]

    elif domain_name == "text_classification":
        from domain.text_classification import TextClassification

        domain = TextClassification()
        data_points = [s.text for s in samples]
        targets = [s.target for s in samples]

    else:
        from domain.chess import Chess

        domain = Chess()
        data_points = [s.board for s in samples]
        targets = [s.evaluation for s in samples]

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        rows = list(
            tqdm(
                ex.map(
                    _compute_row,
                    [features] * len(data_points),
                    data_points,
                    [domain] * len(data_points),
                    chunksize=8,
                ),
                total=len(data_points),
            )
        )

    X = np.array(rows)
    y = np.array(targets)
    return X, y


def compute_feature_in_parallel(feature: Feature, samples: list[Any]) -> list[float]:
    """Compute a feature in parallel for a list of samples."""
    if not samples:
        return []

    with ProcessPoolExecutor(max_workers=os.cpu_count() or 1) as ex:
        return list(
            ex.map(
                _row_for_sample_worker,
                repeat(feature.code),
                repeat(feature.domain),
                samples,
                chunksize=64,
            )
        )


def _row_for_sample_worker(feature_code, feature_domain, sample) -> float:
    feature = Feature(feature_code, feature_domain)
    vals = feature.execute(sample.image)
    return vals[0] if isinstance(vals, list) else vals



def check_feature_worker(code: str, validation_inputs: list, domain) -> None:
    """
    Runs inside a separate process. Raises on failure; returns None on success.
    We reconstruct the Feature from `code` since it has a non-picklable function inside.
    """
    if SILENCE_FEATURE_ERRORS:
        warnings.simplefilter('ignore')
    else:
        # Turns numpy warnings (NaNs, overflow etc) into exceptions.
        warnings.simplefilter('error')

    feature = Feature(code, domain)

    for x in validation_inputs:
        assert np.isfinite(feature.execute(x)).all()
