from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping

import pandas as pd


CLASSIFICATION_STRATEGY = "direct_multiclass_classification"
REGRESSION_STRATEGY = "regression_with_categorization"
RANDOM_STATE = 42

REQUIRED_STATE_KEYS = [
    "X_train",
    "X_test",
    "y_train",
    "y_train_cont",
    "strategy_type",
    "first_level_params",
    "first_level_strategy_type",
    "Q1",
    "Q3",
]

META_STATE_KEYS = [
    "meta_X_train",
    "meta_X_test",
    "meta_feature_names",
    "meta_generation_params",
    "meta_base_models",
    "meta_full_train_models",
    "meta_ready",
    "meta_strategy_type",
]

STRATEGY_LABELS = {
    CLASSIFICATION_STRATEGY: "Пряма багатокласова класифікація",
    REGRESSION_STRATEGY: "Регресія з подальшою категоризацією",
}

MODEL_NAME_LABELS = {
    "RidgeClassifier": "Ridge",
    "RandomForestClassifier": "Random Forest",
    "XGBClassifier": "XGBoost",
    "Ridge": "Ridge",
    "RandomForestRegressor": "Random Forest",
    "XGBRegressor": "XGBoost",
}


@dataclass(frozen=True)
class MetaFeatureContext:
    X_train: pd.DataFrame | object
    X_test: pd.DataFrame | object
    y_train: pd.Series | object | None
    y_train_cont: pd.Series | object | None
    strategy_type: str
    params: Mapping[str, object]
    q1: float | None
    q3: float | None
    target_col: str | None


@dataclass(frozen=True)
class MetaFeatureConfig:
    cv_folds: int
    passthrough: bool


@dataclass(frozen=True)
class MetaFeatureArtifacts:
    meta_X_train: pd.DataFrame
    meta_X_test: pd.DataFrame
    feature_names: List[str]
    base_model_names: List[str]
    full_train_models: Dict[str, object]
    generation_params: Dict[str, object]
