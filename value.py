#!/usr/bin/env python3

from chess import Board, BLACK

from feature_engine import execute_feature


class ValueFunction:
    def predict(self, boards: list[Board]) -> list[float]:
        """
        Predict values for a list of chess positions.
        :param positions: A list of ChessPosition objects.
        :return: A list of predicted values.
        """
        raise NotImplementedError


class RFValueFunction:
    def __init__(self, model, features):
        self.model = model
        self.features = features

    def predict(self, boards: list[Board]) -> list[float]:
        """Value function for the learned model

            This is sequential as is, but the loop over boards is embarrassingly parallel.
        """
        values = []

        for b in boards:
            feature_values = [
                v for feature in self.features for v in execute_feature(feature, b)
            ]
            pred = self.model.predict([feature_values])[0]
            values.append(pred)

        return values

    def __call__(self, boards: list[Board]) -> list[float]:
        preds = self.predict(boards)
        return [p * (-1 if b.turn is BLACK else 1) for p, b in zip(preds, boards)]
