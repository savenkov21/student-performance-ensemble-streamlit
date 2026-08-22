from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


CLASSIFICATION_STRATEGY = "direct_multiclass_classification"
REGRESSION_STRATEGY = "regression_with_categorization"
SINGLE_STRATEGY_REPORT_TYPE = "single_strategy_evaluation"
STATISTICAL_REPORT_TYPE = "classification_statistical_analysis"

STRATEGY_LABELS = {
    CLASSIFICATION_STRATEGY: "Пряма багатокласова класифікація",
    REGRESSION_STRATEGY: "Регресія з подальшою категоризацією",
}

METRIC_NAMES = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced Accuracy",
    "f1_macro": "F1 macro",
    "f1_weighted": "F1 weighted",
}

METRIC_ORDER = ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted"]


@dataclass(frozen=True)
class StatisticalAnalysisConfig:
    confidence_level: float = 0.95
    n_bootstrap: int = 2000
    random_state: int = 42
    alpha: float = 0.05


@dataclass
class StrategyStatisticalInput:
    file_name: str
    strategy_type: str
    strategy_label: str
    target_col: str
    q1: float
    q3: float
    test_set_fingerprint: str
    sample_ids: List[Any]
    y_true: List[Any]
    classes: List[Any]
    models: Dict[str, Dict[str, Any]]


@dataclass
class StatisticalAnalysisContext:
    strategies: List[StrategyStatisticalInput]
    source_signature: str


@dataclass
class StatisticalAnalysisArtifacts:
    report: Dict[str, Any]
    model_rows: List[Dict[str, Any]]
    pairwise_rows: List[Dict[str, Any]]
