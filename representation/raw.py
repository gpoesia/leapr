#!/usr/bin/env python3

"""
Baseline feature learner that returns raw features directly from the input representation
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class RawLearner:
    def __init__(self, max_image_features: int = None, sample_stride: int = 1):
        """
        Args: (for the image domain)
            max_image_features: Max number of features (None = all pixels)
            sample_stride: Sample every Nth pixel
        """
        self.max_image_features = max_image_features
        self.sample_stride = sample_stride

    def learn_features(
        self,
        domain,
        training_set: list[Any],
        validation_set: list[Any],
    ) -> list[str]:
        """Generate raw features based on domain."""

        domain_name = domain.domain_name()
        logger.info(f"Generating raw features for {domain_name}")

        if domain_name == "chess":
            features = self._chess_features()
        elif domain_name == "image_classification":
            if training_set:
                sample = training_set[0]  # ImageSample object
                h, w = sample.image.shape[:2]
                features = self._image_features(h * w)
            else:
                features = self._image_features(784)  # Default MNIST
        else:
            raise ValueError(f"Unknown domain: {domain_name}")

        logger.info(f"Generated {len(features)} raw features")
        return features

    def _chess_features(self) -> list[str]:
        """Generate raw features for chess board squares."""
        features = []
        piece_values = {
            "chess.PAWN": 1,
            "chess.KNIGHT": 3,
            "chess.BISHOP": 3,
            "chess.ROOK": 5,
            "chess.QUEEN": 9,
            "chess.KING": 0,
        }

        for sq in range(64):
            code = f"""def feature(board: chess.Board) -> float:
    "Value at square {sq}"
    p = board.piece_at({sq})
    if not p:
        return 0.0
    vals = {{1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0}}
    return float(vals.get(p.piece_type, 0) * (1 if p.color else -1))"""
            features.append(code)

        return features

    def _image_features(self, n_pixels: int) -> list[str]:
        """Generate raw pixel features for images."""
        features = []

        # Sample pixels based on stride
        pixel_indices = range(0, n_pixels, self.sample_stride)

        # If needed apply limit
        if self.max_image_features is not None:
            pixel_indices = list(pixel_indices)[: self.max_image_features]

        for i in pixel_indices:
            code = f"""def feature(image: np.ndarray) -> float:
    "Pixel {i}"
    return float(image.flat[{i}]) if {i} < image.size else 0.0"""
            features.append(code)

        return features
