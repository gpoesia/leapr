#!/usr/bin/env python3
import logging
import math
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    f1_score,
)
from feature_engine import prepare_supervised_data

logger = logging.getLogger(__name__)


def evaluate_regression_model(model, X, y_true) -> dict[str, float]:
    """Evaluate a regression model."""
    y_pred = model.predict(X)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    rmse = math.sqrt(mse)
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "rho": correlation,
        "y_true": np.array(y_true).tolist(),
        "y_pred": np.array(y_pred).tolist(),
    }


def evaluate_classification_model(model, X, y_true) -> dict[str, float]:
    """Evaluate a classification model."""
    y_pred = model.predict(X)
    accuracy = accuracy_score(y_true, y_pred)
    if len(set(y_true)) == 2:
        f1 = f1_score(y_true, y_pred, average="binary")  # binary for 2-class problems
    else:
        f1 = None

    return {
        "accuracy": accuracy,
        "f1": f1,
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
    }


def prepare_train_valid_split(
    features: list[str],
    train_positions: list,
    valid_positions: list,
    domain_name: str = "chess",
):
    """Prepare training and validation data splits."""
    X, y = prepare_supervised_data(
        features, train_positions + valid_positions, domain_name
    )
    X_train, y_train = X[: len(train_positions)], y[: len(train_positions)]
    X_valid, y_valid = X[len(train_positions) :], y[len(train_positions) :]
    return X_train, y_train, X_valid, y_valid
