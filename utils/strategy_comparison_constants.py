from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd


CLASSIFICATION_STRATEGY = "direct_multiclass_classification"
REGRESSION_STRATEGY = "regression_with_categorization"
EXPECTED_REPORT_TYPE = "single_strategy_evaluation"

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
    "SoftVoting": "Soft Voting",
    "Voting": "Voting",
    "Stacking": "Stacking",
}

PREDICTION_REQUIRED_KEYS = [
    "split_ready",
    "preprocessor",
    "feature_cols",
    "X_raw",
    "first_level_models",
    "first_level_results",
    "second_level_models",
    "second_level_results",
    "second_level_params",
    "meta_full_train_models",
    "meta_generation_params",
    "strategy_type",
    "Q1",
    "Q3",
]

CLASSIFICATION_METRICS = [
    ("Accuracy", "Accuracy"),
    ("Balanced Accuracy", "Balanced_accuracy"),
    ("F1 weighted", "F1_weighted"),
]

REGRESSION_METRICS = [
    ("MAE", "MAE"),
    ("MSE", "MSE"),
    ("RMSE", "RMSE"),
    ("R²", "R2"),
    ("Accuracy", "Accuracy_after_discretization"),
    ("Balanced Accuracy", "Balanced_accuracy_after_discretization"),
    ("F1 weighted", "F1_weighted_after_discretization"),
]


@dataclass
class StrategyReport:
    strategy_type: str
    strategy_label: str
    file_name: str
    target_col: str
    q1: Any
    q3: Any
    first_level_results: Dict[str, Dict[str, Any]]
    second_level_results: Dict[str, Dict[str, Any]]
    comparison_table: List[Dict[str, Any]]
    first_level_params: Dict[str, Any]
    meta_generation_params: Dict[str, Any]
    second_level_params: Dict[str, Any]
    raw_payload: Dict[str, Any]


@dataclass
class LivePredictionContext:
    strategy_type: str
    preprocessor: Any
    feature_cols: List[str]
    x_raw: pd.DataFrame
    first_level_models: Dict[str, Any]
    first_level_results: Dict[str, Dict[str, Any]]
    second_level_models: Dict[str, Any]
    second_level_results: Dict[str, Dict[str, Any]]
    second_level_params: Dict[str, Any]
    meta_full_train_models: Dict[str, Any]
    meta_generation_params: Dict[str, Any]
    q1: float
    q3: float
    feature_names_after_preprocessing: Optional[List[str]] = None
