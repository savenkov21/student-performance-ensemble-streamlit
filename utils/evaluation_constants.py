from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd


CLASSIFICATION_STRATEGY = "direct_multiclass_classification"
REGRESSION_STRATEGY = "regression_with_categorization"

STRATEGY_LABELS = {
    CLASSIFICATION_STRATEGY: "Пряма багатокласова класифікація",
    REGRESSION_STRATEGY: "Регресія з подальшою категоризацією",
}

STRATEGY_EXPORT_SLUGS = {
    CLASSIFICATION_STRATEGY: "classification",
    REGRESSION_STRATEGY: "regression",
}

PRIMARY_METRIC_BY_STRATEGY = {
    CLASSIFICATION_STRATEGY: "F1 weighted",
    REGRESSION_STRATEGY: "F1 weighted",
}

REQUIRED_STATE_KEYS = [
    "strategy_type",
    "first_level_results",
    "first_level_strategy_type",
    "first_level_predictions_cls",
    "second_level_results",
    "second_level_strategy_type",
    "second_level_predictions_cls",
    "y_test_cls",
]


@dataclass
class EvaluationContext:
    strategy_type: str
    target_col: str
    q1: Any
    q3: Any
    meta_ready: bool
    first_level_results: Dict[str, Dict[str, Any]]
    second_level_results: Dict[str, Dict[str, Any]]
    first_level_params: Dict[str, Any]
    meta_generation_params: Dict[str, Any]
    second_level_params: Dict[str, Any]
    y_test_cls: Any
    test_sample_ids: Any
    test_features: Any
    first_level_predictions_cls: Dict[str, Any]
    second_level_predictions_cls: Dict[str, Any]


@dataclass
class EvaluationArtifacts:
    comparison_df: pd.DataFrame
    final_report: Dict[str, Any]
