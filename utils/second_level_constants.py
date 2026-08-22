from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


CLASSIFICATION_STRATEGY = "direct_multiclass_classification"
REGRESSION_STRATEGY = "regression_with_categorization"
RANDOM_STATE = 42
SOFT_VOTING_NAME = "SoftVoting"
VOTING_NAME = "Voting"
STACKING_NAME = "Stacking"

REQUIRED_META_KEYS = [
    "meta_X_train",
    "meta_X_test",
    "strategy_type",
    "meta_ready",
    "meta_strategy_type",
    "Q1",
    "Q3",
]

SECOND_LEVEL_STATE_KEYS = [
    "second_level_models",
    "second_level_results",
    "second_level_predictions",
    "second_level_predictions_cls",
    "second_level_params",
    "second_level_strategy_type",
]

STRATEGY_LABELS = {
    CLASSIFICATION_STRATEGY: "Пряма багатокласова класифікація",
    REGRESSION_STRATEGY: "Регресія з подальшою категоризацією",
}


@dataclass
class SecondLevelContext:
    meta_X_train: pd.DataFrame
    meta_X_test: pd.DataFrame
    strategy_type: str
    q1: float
    q3: float
    target_col: str | None
    y_train_cls: pd.Series | None = None
    y_test_cls: pd.Series | None = None
    y_train_cont: pd.Series | None = None
    y_test_cont: pd.Series | None = None


@dataclass
class SecondLevelConfig:
    classification_weights: list[float] | None
    regression_weights: list[float] | None


@dataclass
class SecondLevelArtifacts:
    models: dict[str, Any]
    results: dict[str, dict[str, Any]]
    predictions: dict[str, Any]
    predictions_cls: dict[str, np.ndarray]
    params: dict[str, dict[str, Any]]
