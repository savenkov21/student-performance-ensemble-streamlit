from __future__ import annotations

import hashlib
import json
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from utils.evaluation_constants import (
    CLASSIFICATION_STRATEGY,
    PRIMARY_METRIC_BY_STRATEGY,
    REGRESSION_STRATEGY,
    STRATEGY_EXPORT_SLUGS,
    STRATEGY_LABELS,
    EvaluationArtifacts,
    EvaluationContext,
)


def _json_safe_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _json_safe_list(values: Any) -> List[Any]:
    if isinstance(values, (pd.Series, pd.Index)):
        values = values.tolist()
    elif isinstance(values, np.ndarray):
        values = values.tolist()
    return [_json_safe_scalar(value) for value in list(values)]


def results_to_json_bytes(results_dict: Dict[str, Any]) -> BytesIO:
    json_bytes = BytesIO()
    json_bytes.write(json.dumps(results_dict, indent=4, ensure_ascii=False).encode("utf-8"))
    json_bytes.seek(0)
    return json_bytes


def strategy_label(strategy_type: str) -> str:
    return STRATEGY_LABELS.get(strategy_type, strategy_type)


def get_primary_metric(strategy_type: str) -> str:
    return PRIMARY_METRIC_BY_STRATEGY.get(strategy_type, "F1 weighted")


def strategy_export_slug(strategy_type: str) -> str:
    return STRATEGY_EXPORT_SLUGS.get(strategy_type, "strategy")


def build_export_filename(strategy_type: str) -> str:
    kyiv_now = datetime.now(ZoneInfo("Europe/Kyiv"))
    date_part = kyiv_now.strftime("%Y-%m-%d")
    return f"{strategy_export_slug(strategy_type)}_strategy_results_{date_part}.json"


def metric_value(value: Any) -> str:
    return f"{float(value):.4f}" if isinstance(value, (int, float)) else str(value)


def get_result_value(result: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in result and result[key] is not None:
            return result[key]
    if default is not None:
        return default
    raise KeyError(keys[0] if keys else "missing_key")


def classification_row(model_name: str, result: Dict[str, Any], level: str, family: str) -> Dict[str, Any]:
    return {
        "Strategy": "Direct multiclass classification",
        "Model": model_name,
        "Level": level,
        "Family": family,
        "Accuracy": float(get_result_value(result, "Accuracy", default=0.0)),
        "Balanced Accuracy": float(get_result_value(result, "Balanced_accuracy", default=0.0)),
        "F1 weighted": float(get_result_value(result, "F1_weighted", default=0.0)),
    }


def regression_row(model_name: str, result: Dict[str, Any], level: str, family: str) -> Dict[str, Any]:
    return {
        "Strategy": "Regression with categorization",
        "Model": model_name,
        "Level": level,
        "Family": family,
        "MAE": float(get_result_value(result, "MAE", default=0.0)),
        "MSE": float(get_result_value(result, "MSE", default=0.0)),
        "RMSE": float(get_result_value(result, "RMSE", default=0.0)),
        "R2": float(get_result_value(result, "R2", "R²", default=0.0)),
        "Accuracy": float(
            get_result_value(result, "Accuracy", "Accuracy_after_discretization", default=0.0)
        ),
        "Balanced Accuracy": float(
            get_result_value(
                result,
                "Balanced_accuracy",
                "Balanced_accuracy_after_discretization",
                "Balanced_accuracyter",
                default=0.0,
            )
        ),
        "F1 weighted": float(
            get_result_value(result, "F1_weighted", "F1_weighted_after_discretization", default=0.0)
        ),
    }


def build_comparison_dataframe(
    strategy_type: str,
    first_level_results: Dict[str, Dict[str, Any]],
    second_level_results: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    if strategy_type == CLASSIFICATION_STRATEGY:
        for model_name, result in first_level_results.items():
            rows.append(classification_row(model_name, result, "Перший рівень", "Базова модель"))
        for model_name, result in second_level_results.items():
            rows.append(classification_row(model_name, result, "Другий рівень", "Ансамбль"))
    else:
        for model_name, result in first_level_results.items():
            rows.append(regression_row(model_name, result, "Перший рівень", "Базова модель"))
        for model_name, result in second_level_results.items():
            rows.append(regression_row(model_name, result, "Другий рівень", "Ансамбль"))

    return pd.DataFrame(rows)


def _build_test_set_fingerprint(
    sample_ids: List[Any],
    y_true: List[Any],
    target_col: str,
    q1: Any,
    q3: Any,
    test_features: Any,
) -> tuple[str, List[str]]:
    feature_columns: List[str] = []
    features_payload: Any = None
    if isinstance(test_features, pd.DataFrame):
        feature_columns = [str(column) for column in test_features.columns]
        normalized = test_features.copy()
        normalized.columns = feature_columns
        features_payload = normalized.to_json(
            orient="split",
            date_format="iso",
            double_precision=15,
            force_ascii=False,
        )
    elif test_features is not None:
        features_payload = str(test_features)

    canonical = {
        "sample_ids": sample_ids,
        "y_true": y_true,
        "target_col": _json_safe_scalar(target_col),
        "Q1": _json_safe_scalar(q1),
        "Q3": _json_safe_scalar(q3),
        "feature_columns": feature_columns,
        "test_features": features_payload,
    }
    raw = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest(), feature_columns


def build_statistical_data(context: EvaluationContext) -> Dict[str, Any]:
    y_true = _json_safe_list(context.y_test_cls)
    sample_ids = _json_safe_list(context.test_sample_ids)

    if not sample_ids:
        sample_ids = list(range(len(y_true)))
    if len(sample_ids) != len(y_true):
        raise ValueError("Кількість ідентифікаторів test set не збігається з кількістю істинних класів.")

    classes = sorted({_json_safe_scalar(value) for value in y_true}, key=lambda value: str(value))
    fingerprint, feature_columns = _build_test_set_fingerprint(
        sample_ids=sample_ids,
        y_true=y_true,
        target_col=context.target_col,
        q1=context.q1,
        q3=context.q3,
        test_features=context.test_features,
    )

    models: Dict[str, Dict[str, Any]] = {}
    for level, family, results, predictions in [
        ("Перший рівень", "Базова модель", context.first_level_results, context.first_level_predictions_cls),
        ("Другий рівень", "Ансамбль", context.second_level_results, context.second_level_predictions_cls),
    ]:
        for model_name, result in results.items():
            if model_name not in predictions:
                raise ValueError(f"Для моделі '{model_name}' відсутні покрокові класифікаційні прогнози.")
            y_pred = _json_safe_list(predictions[model_name])
            if len(y_pred) != len(y_true):
                raise ValueError(
                    f"Для моделі '{model_name}' довжина прогнозів ({len(y_pred)}) "
                    f"не збігається з test set ({len(y_true)})."
                )
            models[model_name] = {
                "level": level,
                "family": family,
                "point_metrics_from_evaluation": {
                    "accuracy": float(get_result_value(result, "Accuracy", "Accuracy_after_discretization", default=0.0)),
                    "balanced_accuracy": float(
                        get_result_value(
                            result,
                            "Balanced_accuracy",
                            "Balanced_accuracy_after_discretization",
                            "Balanced_accuracyter",
                            default=0.0,
                        )
                    ),
                    "f1_weighted": float(
                        get_result_value(
                            result,
                            "F1_weighted",
                            "F1_weighted_after_discretization",
                            default=0.0,
                        )
                    ),
                },
                "y_pred": y_pred,
            }

    return {
        "schema_version": 1,
        "test_set": {
            "fingerprint": fingerprint,
            "n_samples": len(y_true),
            "sample_ids": sample_ids,
            "y_true": y_true,
            "classes": classes,
            "feature_columns": feature_columns,
        },
        "models": models,
    }


def build_final_report(
    context: EvaluationContext,
    comparison_df: pd.DataFrame,
) -> Dict[str, Any]:
    return {
        "report_type": "single_strategy_evaluation",
        "schema_version": 3,
        "strategy_type": context.strategy_type,
        "strategy_label": strategy_label(context.strategy_type),
        "target_col": context.target_col,
        "Q1": _json_safe_scalar(context.q1),
        "Q3": _json_safe_scalar(context.q3),
        "first_level_params": context.first_level_params,
        "meta_generation_params": context.meta_generation_params,
        "second_level_params": context.second_level_params,
        "first_level_results": context.first_level_results,
        "second_level_results": context.second_level_results,
        "comparison_table": comparison_df.to_dict(orient="records"),
        "statistical_data": build_statistical_data(context),
    }


def build_artifacts(context: EvaluationContext) -> EvaluationArtifacts:
    comparison_df = build_comparison_dataframe(
        strategy_type=context.strategy_type,
        first_level_results=context.first_level_results,
        second_level_results=context.second_level_results,
    )
    final_report = build_final_report(context, comparison_df)
    return EvaluationArtifacts(comparison_df=comparison_df, final_report=final_report)


def prepare_metric_comparison_df(comparison_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    result_df = comparison_df.copy()
    if metric in result_df.columns:
        result_df = result_df.sort_values(metric, ascending=False).reset_index(drop=True)
    return result_df


def classification_metric_rows(result: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Метрика": "Accuracy", "Значення": float(result["Accuracy"])},
            {"Метрика": "Balanced Accuracy", "Значення": float(result["Balanced_accuracy"])},
            {"Метрика": "F1 weighted", "Значення": float(result["F1_weighted"])},
        ]
    )


def ordered_second_level_names(
    strategy_type: str,
    second_level_results: Dict[str, Dict[str, Any]],
) -> List[str]:
    preferred = ["SoftVoting", "Stacking"] if strategy_type == CLASSIFICATION_STRATEGY else ["Voting", "Stacking"]
    existing = [name for name in preferred if name in second_level_results]
    remaining = [name for name in second_level_results.keys() if name not in existing]
    return existing + remaining
