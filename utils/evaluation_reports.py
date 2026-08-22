from __future__ import annotations

import ast
import re
from typing import Any, Dict, List

import pandas as pd
import streamlit as st


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


def normalize_report_df(report_df: pd.DataFrame) -> pd.DataFrame:
    if report_df.empty:
        return report_df

    rename_map = {
        "index": "Клас / середнє",
        "precision": "Precision",
        "recall": "Recall",
        "f1-score": "F1-score",
        "support": "Support",
    }
    report_df = report_df.rename(columns=rename_map)

    preferred_columns = ["Клас / середнє", "Precision", "Recall", "F1-score", "Support"]
    existing = [col for col in preferred_columns if col in report_df.columns]
    remaining = [col for col in report_df.columns if col not in existing]
    return report_df[existing + remaining]


def report_to_dataframe(report_value: Any) -> pd.DataFrame:
    if report_value is None:
        return pd.DataFrame()

    if isinstance(report_value, dict):
        try:
            return normalize_report_df(pd.DataFrame(report_value).T.reset_index())
        except Exception:
            return pd.DataFrame()

    if isinstance(report_value, str):
        stripped = report_value.strip()
        if not stripped:
            return pd.DataFrame()

        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, dict):
                    return report_to_dataframe(parsed)
            except Exception:
                pass

        return normalize_report_df(parse_classification_report_text(stripped))

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
