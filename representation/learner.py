#!/usr/bin/env python3

from chess_position import ChessPosition
from evaluation import PolicyEvaluator


class RepresentationLearner:
    """
    A Representation Learner takes a training set of chess positions and
    generates a set of features, each of which must be the code of a
    Python function of type Board -> float.
    """
    def learn_features(
        self,
        _training_positions: list[ChessPosition],
        _validation_positions: list[ChessPosition],
        _evaluation_positions: list[ChessPosition],
        _evaluator: PolicyEvaluator,
    ) -> list[str]:
        raise NotImplementedError
