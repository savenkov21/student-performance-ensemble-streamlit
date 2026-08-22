from dataclasses import dataclass
from typing import Any


STRATEGY_LABELS = {
    "direct_multiclass_classification": "Пряма багатокласова класифікація",
    "regression_with_categorization": "Регресія з подальшою категоризацією",
}

DEFAULT_PREPROCESSING_CONFIG = {
    "scaler_type": "StandardScaler",
    "numeric_imputer_strategy": "mean",
    "categorical_imputer_strategy": "most_frequent",
    "onehot_drop": "if_binary",
    "test_size": 0.30,
    "random_state": 42,
}

NUMERIC_IMPUTER_LABELS = {
    "Середнє значення": "mean",
    "Медіана": "median",
}
NUMERIC_IMPUTER_LABELS_REVERSE = {v: k for k, v in NUMERIC_IMPUTER_LABELS.items()}

CATEGORICAL_IMPUTER_LABELS = {
    "Найчастіше значення": "most_frequent",
    "Окреме службове значення": "constant",
}
CATEGORICAL_IMPUTER_LABELS_REVERSE = {v: k for k, v in CATEGORICAL_IMPUTER_LABELS.items()}

ONEHOT_ENCODING_LABELS = {
    "Повне кодування": "none",
    "Скорочене кодування для бінарних ознак": "if_binary",
}
ONEHOT_ENCODING_LABELS_REVERSE = {v: k for k, v in ONEHOT_ENCODING_LABELS.items()}

SPLIT_STATE_KEYS = [
    # module_2
    "strategy_type",
    "target_col",
    "feature_cols",
    "feature_names_after_preprocessing",
    "preprocessing_config",
    "preprocessor",
    "X_raw",
    "X_train_raw",
    "X_test_raw",
    "X",
    "y",
    "X_train",
    "X_test",
    "y_train",
    "y_test",
    "y_cont",
    "y_train_cont",
    "y_test_cont",
    "y_train_cls",
    "y_test_cls",
    "Q1",
    "Q3",
    "test_size",
    "random_state",
    "stratify_used",
    "split_ready",
    # module_4
    "first_level_models",
    "first_level_results",
    "first_level_predictions",
    "first_level_predictions_cls",
    "first_level_params",
    "first_level_strategy_type",
    "optimization_results",
    "optimization_mode",
    # module_5
    "meta_X_train",
    "meta_X_test",
    "meta_feature_names",
    "meta_generation_params",
    "meta_base_models",
    "meta_full_train_models",
    "meta_ready",
    "meta_strategy_type",
    # module_6
    "second_level_models",
    "second_level_model",
    "second_level_results",
    "second_level_predictions",
    "second_level_predictions_cls",
    "second_level_params",
    "second_level_strategy_type",
]


@dataclass
class SplitConfig:
    target_col: str
    strategy_type: str
    scaler_type: str
    numeric_imputer_strategy: str
    categorical_imputer_strategy: str
    onehot_drop: str
    test_size: float
    random_state: int


@dataclass
class SplitArtifacts:
    strategy_type: str
    target_col: str
    feature_cols: list[str]
    feature_names_after_preprocessing: list[str]
    preprocessing_config: dict[str, Any]
    preprocessor: Any
    X_raw: Any
    X_train_raw: Any
    X_test_raw: Any
    X_train: Any
    X_test: Any
    y_active: Any
    y_train: Any
    y_test: Any
    y_cont: Any
    y_train_cont: Any
    y_test_cont: Any
    y_train_cls: Any
    y_test_cls: Any
    q1: float
    q3: float
    test_size: float
    random_state: int
    stratify_used: bool
