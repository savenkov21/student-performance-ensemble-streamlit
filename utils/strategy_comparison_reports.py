from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from utils.strategy_comparison_constants import (
    CLASSIFICATION_STRATEGY,
    EXPECTED_REPORT_TYPE,
    REGRESSION_STRATEGY,
    STRATEGY_LABELS,
    StrategyReport,
)


def strategy_label(strategy_type: str) -> str:
    return STRATEGY_LABELS.get(strategy_type, strategy_type)


def uploaded_name(uploaded_file) -> str:
    return getattr(uploaded_file, "name", "uploaded_report.json")


def parse_uploaded_json(uploaded_file) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if uploaded_file is None:
        return None, None

    try:
        raw = uploaded_file.getvalue().decode("utf-8")
    except Exception:
        return None, f"Файл '{uploaded_name(uploaded_file)}' не вдалося прочитати як UTF-8 JSON."

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"Файл '{uploaded_name(uploaded_file)}' не є коректним JSON: {exc}."

    if not isinstance(payload, dict):
        return None, f"Файл '{uploaded_name(uploaded_file)}' повинен містити JSON-об'єкт, а не список або інший тип даних."

    return payload, None


def validate_report_payload(payload: Dict[str, Any], expected_strategy: Optional[str] = None) -> Optional[str]:
    required_keys = [
        "strategy_type",
        "strategy_label",
        "first_level_results",
        "second_level_results",
        "comparison_table",
    ]
    missing = [key for key in required_keys if key not in payload]
    if missing:
        return f"У JSON відсутні обов'язкові поля: {', '.join(missing)}."

    report_type = payload.get("report_type")
    if report_type is not None and report_type != EXPECTED_REPORT_TYPE:
        return f"Непідтримуваний тип звіту '{report_type}'. Очікується '{EXPECTED_REPORT_TYPE}'."

    strategy_type = payload.get("strategy_type")
    if expected_strategy is not None and strategy_type != expected_strategy:
        return f"Файл містить стратегію '{strategy_type}', але для цього поля очікується '{expected_strategy}'."

    if strategy_type not in {CLASSIFICATION_STRATEGY, REGRESSION_STRATEGY}:
        return f"Непідтримуване значення поля 'strategy_type': '{strategy_type}'."

    if not isinstance(payload.get("first_level_results"), dict):
        return "Поле 'first_level_results' повинно бути словником результатів базових моделей."
    if not isinstance(payload.get("second_level_results"), dict):
        return "Поле 'second_level_results' повинно бути словником результатів ансамблів другого рівня."
    if not isinstance(payload.get("comparison_table"), list):
        return "Поле 'comparison_table' повинно бути списком рядків підсумкової таблиці."

    return None


def build_strategy_report(payload: Dict[str, Any], file_name: str) -> StrategyReport:
    strategy_type = payload["strategy_type"]
    return StrategyReport(
        strategy_type=strategy_type,
        strategy_label=payload.get("strategy_label", strategy_label(strategy_type)),
        file_name=file_name,
        target_col=payload.get("target_col", "—"),
        q1=payload.get("Q1", "—"),
        q3=payload.get("Q3", "—"),
        first_level_results=payload.get("first_level_results", {}),
        second_level_results=payload.get("second_level_results", {}),
        comparison_table=payload.get("comparison_table", []),
        first_level_params=payload.get("first_level_params", {}),
        meta_generation_params=payload.get("meta_generation_params", {}),
        second_level_params=payload.get("second_level_params", {}),
        raw_payload=payload,
    )


def parse_classification_report_text(report_text: str) -> pd.DataFrame:
    if not report_text or not isinstance(report_text, str):
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for raw_line in report_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("precision") or line.startswith("accuracy"):
            continue

        parts = re.split(r"\s{2,}", line)
        if len(parts) < 2:
            continue

        label = parts[0]
        numeric_parts = parts[1:]
        if len(numeric_parts) >= 4:
            try:
                rows.append(
                    {
                        "Клас / середнє": label,
                        "Precision": float(numeric_parts[0]),
                        "Recall": float(numeric_parts[1]),
                        "F1-score": float(numeric_parts[2]),
                        "Support": int(float(numeric_parts[3])),
                    }
                )
            except ValueError:
                continue

    return pd.DataFrame(rows)


def report_to_dataframe(report_value: Any) -> pd.DataFrame:
    if isinstance(report_value, dict):
        try:
            return pd.DataFrame(report_value).T.reset_index().rename(columns={"index": "Клас / середнє"})
        except Exception:
            return pd.DataFrame()
    if isinstance(report_value, str):
        stripped = report_value.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, dict):
                    return report_to_dataframe(parsed)
            except Exception:
                pass
        return parse_classification_report_text(report_value)
    return pd.DataFrame()


def render_classification_report(report_value: Any, title: str) -> None:
    if report_value is None:
        return

    st.markdown(f"**{title}**")
    report_df = report_to_dataframe(report_value)
    if not report_df.empty:
        st.dataframe(report_df, use_container_width=True, hide_index=True)
        return

    if isinstance(report_value, str):
        st.code(report_value, language="text")
        return

    st.caption("Не вдалося відобразити classification report у табличному вигляді.")
