from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional

import pandas as pd

from utils.strategy_comparison_constants import (
    CLASSIFICATION_METRICS,
    CLASSIFICATION_STRATEGY,
    MODEL_DISPLAY_NAMES,
    REGRESSION_METRICS,
    REGRESSION_STRATEGY,
    STRATEGY_LABELS,
    StrategyReport,
)


def strategy_label(strategy_type: str) -> str:
    return STRATEGY_LABELS.get(strategy_type, strategy_type)


def metric_value(value: Any, digits: int = 4) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def model_display_name(model_name: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model_name, model_name)


def comparison_df(report: StrategyReport) -> pd.DataFrame:
    df = pd.DataFrame(report.comparison_table)
    if df.empty:
        return df

    if "F1 weighted" in df.columns:
        df = df.sort_values("F1 weighted", ascending=False).reset_index(drop=True)

    columns_to_drop = ["Strategy"]
    return df.drop(columns=[col for col in columns_to_drop if col in df.columns])


def cross_strategy_models_df(
    classification_report: StrategyReport,
    regression_report: StrategyReport,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for report in [classification_report, regression_report]:
        for result in report.comparison_table:
            rows.append(
                {
                    "Стратегія": report.strategy_label,
                    "Модель": model_display_name(str(result.get("Model", "—"))),
                    "Рівень": result.get("Level", "—"),
                    "Тип": result.get("Family", "—"),
                    "Accuracy": safe_float(result.get("Accuracy")),
                    "Balanced Accuracy": safe_float(result.get("Balanced Accuracy")),
                    "F1 weighted": safe_float(result.get("F1 weighted")),
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty and "F1 weighted" in df.columns:
        df = df.sort_values(["Стратегія", "F1 weighted"], ascending=[True, False]).reset_index(drop=True)
    return df


def metric_df_from_mapping(result: Dict[str, Any], mapping: List[Tuple[str, str]]) -> pd.DataFrame:
    rows = [
        {"Метрика": label, "Значення": safe_float(result.get(key))}
        for label, key in mapping
        if key in result
    ]
    return pd.DataFrame(rows)


def classification_metric_df(result: Dict[str, Any]) -> pd.DataFrame:
    return metric_df_from_mapping(result, CLASSIFICATION_METRICS)


def regression_metric_df(result: Dict[str, Any]) -> pd.DataFrame:
    return metric_df_from_mapping(result, REGRESSION_METRICS)


def ordered_second_level_names(strategy_type: str, second_level_results: Dict[str, Dict[str, Any]]) -> List[str]:
    preferred = ["SoftVoting", "Stacking"] if strategy_type == CLASSIFICATION_STRATEGY else ["Voting", "Stacking"]
    existing = [name for name in preferred if name in second_level_results]
    remaining = [name for name in second_level_results.keys() if name not in existing]
    return existing + remaining
