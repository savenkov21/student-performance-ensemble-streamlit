from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


CLASSIFICATION_STRATEGY = "direct_multiclass_classification"
REGRESSION_STRATEGY = "regression_with_categorization"

FIRST_LEVEL_STATE_KEYS = [
    "first_level_models",
    "first_level_results",
    "first_level_predictions",
    "first_level_predictions_cls",
    "first_level_params",
    "first_level_strategy_type",
    "optimization_results",
    "optimization_mode",
]

REQUIRED_SPLIT_KEYS = [
    "X_train",
    "X_test",
    "y_train",
    "y_test",
    "strategy_type",
    "split_ready",
    "Q1",
    "Q3",
    "y_train_cont",
    "y_test_cont",
]

STRATEGY_LABELS = {
    CLASSIFICATION_STRATEGY: "Пряма багатокласова класифікація",
    REGRESSION_STRATEGY: "Регресія з подальшою категоризацією",
}

MODEL_DISPLAY_NAMES = {
    "Ridge": "Ridge",
    "RandomForest": "Random Forest",
    "XGBoost": "XGBoost",
    "RidgeClassifier": "Ridge Classifier",
    "RandomForestClassifier": "Random Forest Classifier",
    "XGBClassifier": "XGBoost Classifier",
    "RandomForestRegressor": "Random Forest Regressor",
    "XGBRegressor": "XGBoost Regressor",
}

PRIMARY_METRIC_BY_STRATEGY = {
    CLASSIFICATION_STRATEGY: "F1 weighted",
    REGRESSION_STRATEGY: "F1 weighted",
}


@dataclass(frozen=True)
class TrainingContext:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    strategy_type: str
    q1: float
    q3: float
    y_train: pd.Series
    y_test: pd.Series
    y_train_cont: pd.Series
    y_test_cont: pd.Series
    target_col: str


@dataclass(frozen=True)
class OptimizationConfig:
    init_points: int
    n_iter: int


@dataclass(frozen=True)
class SelectedHyperparameters:
    ridge_alpha: float
    rf_params: dict[str, Any]
    xgb_params: dict[str, Any]


@dataclass(frozen=True)
class TrainingArtifacts:
    models: dict[str, Any]
    results: dict[str, dict[str, Any]]
    predictions: dict[str, np.ndarray]
    predictions_cls: dict[str, np.ndarray]
    params: dict[str, Any]
