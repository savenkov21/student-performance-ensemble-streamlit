from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


CLASSIFICATION_STRATEGY = "direct_multiclass_classification"
REGRESSION_STRATEGY = "regression_with_categorization"
EXPECTED_REPORT_TYPE = "single_strategy_evaluation"


@dataclass
class StrategyReport:
    strategy_type: str
    strategy_label: str
    file_name: str
    target_col: str
    q1: Any
    q3: Any
    best_model: Dict[str, Any]
    first_level_results: Dict[str, Dict[str, Any]]
    second_level_results: Dict[str, Dict[str, Any]]
    comparison_table: List[Dict[str, Any]]
    first_level_params: Dict[str, Any]
    meta_generation_params: Dict[str, Any]
    second_level_params: Dict[str, Any]
    raw_payload: Dict[str, Any]


# =========================================================
# Basic helpers
# =========================================================
def _strategy_label(strategy_type: str) -> str:
    mapping = {
        CLASSIFICATION_STRATEGY: "Пряма багатокласова класифікація",
        REGRESSION_STRATEGY: "Регресія з подальшою категоризацією",
    }
    return mapping.get(strategy_type, strategy_type)


def _metric_value(value: Any, digits: int = 4) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _model_display_name(model_name: str) -> str:
    labels = {
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
    return labels.get(model_name, model_name)


# =========================================================
# JSON validation / parsing
# =========================================================
def _uploaded_name(uploaded_file) -> str:
    return getattr(uploaded_file, "name", "uploaded_report.json")


def _parse_uploaded_json(uploaded_file) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if uploaded_file is None:
        return None, None

    try:
        raw = uploaded_file.getvalue().decode("utf-8")
    except Exception:
        return None, f"Файл '{_uploaded_name(uploaded_file)}' не вдалося прочитати як UTF-8 JSON."

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"Файл '{_uploaded_name(uploaded_file)}' не є коректним JSON: {exc}."

    if not isinstance(payload, dict):
        return None, f"Файл '{_uploaded_name(uploaded_file)}' повинен містити JSON-об'єкт, а не список або інший тип даних."

    return payload, None


def _validate_report_payload(payload: Dict[str, Any], expected_strategy: Optional[str] = None) -> Optional[str]:
    required_keys = [
        "strategy_type",
        "strategy_label",
        "best_model",
        "first_level_results",
        "second_level_results",
        "comparison_table",
    ]
    missing = [key for key in required_keys if key not in payload]
    if missing:
        return f"У JSON відсутні обов'язкові поля: {', '.join(missing)}."

    report_type = payload.get("report_type")
    if report_type is not None and report_type != EXPECTED_REPORT_TYPE:
        return (
            f"Непідтримуваний тип звіту '{report_type}'. Очікується '{EXPECTED_REPORT_TYPE}'."
        )

    strategy_type = payload.get("strategy_type")
    if expected_strategy is not None and strategy_type != expected_strategy:
        return (
            f"Файл містить стратегію '{strategy_type}', але для цього поля очікується '{expected_strategy}'."
        )

    if strategy_type not in {CLASSIFICATION_STRATEGY, REGRESSION_STRATEGY}:
        return (
            f"Непідтримуване значення поля 'strategy_type': '{strategy_type}'."
        )

    if not isinstance(payload.get("best_model"), dict):
        return "Поле 'best_model' повинно бути словником із характеристиками найкращої моделі."
    if not isinstance(payload.get("first_level_results"), dict):
        return "Поле 'first_level_results' повинно бути словником результатів базових моделей."
    if not isinstance(payload.get("second_level_results"), dict):
        return "Поле 'second_level_results' повинно бути словником результатів ансамблів другого рівня."
    if not isinstance(payload.get("comparison_table"), list):
        return "Поле 'comparison_table' повинно бути списком рядків підсумкової таблиці."

    return None


def _build_strategy_report(payload: Dict[str, Any], file_name: str) -> StrategyReport:
    strategy_type = payload["strategy_type"]
    return StrategyReport(
        strategy_type=strategy_type,
        strategy_label=payload.get("strategy_label", _strategy_label(strategy_type)),
        file_name=file_name,
        target_col=payload.get("target_col", "—"),
        q1=payload.get("Q1", "—"),
        q3=payload.get("Q3", "—"),
        best_model=payload.get("best_model", {}),
        first_level_results=payload.get("first_level_results", {}),
        second_level_results=payload.get("second_level_results", {}),
        comparison_table=payload.get("comparison_table", []),
        first_level_params=payload.get("first_level_params", {}),
        meta_generation_params=payload.get("meta_generation_params", {}),
        second_level_params=payload.get("second_level_params", {}),
        raw_payload=payload,
    )


# =========================================================
# Styling
# =========================================================
def _inject_page_style() -> None:
    st.markdown(
        """
        <style>
        .m8-hero {
            padding: 1rem 1.15rem;
            border: 1px solid rgba(120,120,120,0.20);
            border-radius: 14px;
            background: linear-gradient(180deg, rgba(70,130,180,0.08), rgba(255,255,255,0.02));
            margin-bottom: 0.8rem;
        }
        .m8-section-title {
            margin-top: 0.55rem;
            margin-bottom: 0.15rem;
            font-weight: 700;
            font-size: 1.05rem;
        }
        .m8-subtle {
            color: rgba(120,120,120,0.95);
            font-size: 0.95rem;
            margin-bottom: 0.25rem;
        }
        .m8-card {
            border: 1px solid rgba(120,120,120,0.18);
            border-radius: 14px;
            padding: 0.85rem 0.95rem;
            background: rgba(255,255,255,0.02);
            min-height: 108px;
        }
        .m8-card-kicker {
            font-size: 0.82rem;
            color: rgba(120,120,120,0.95);
            margin-bottom: 0.35rem;
        }
        .m8-card-value {
            font-size: 1.18rem;
            font-weight: 700;
            line-height: 1.3;
            margin-bottom: 0.3rem;
            word-break: break-word;
        }
        .m8-card-note {
            font-size: 0.86rem;
            color: rgba(120,120,120,0.92);
        }
        .m8-note {
            border-left: 4px solid rgba(70,130,180,0.55);
            padding: 0.75rem 0.9rem;
            background: rgba(70,130,180,0.06);
            border-radius: 10px;
            margin-top: 0.4rem;
            margin-bottom: 0.35rem;
        }
        .m8-pill {
            display: inline-block;
            padding: 0.2rem 0.55rem;
            border-radius: 999px;
            border: 1px solid rgba(120,120,120,0.25);
            font-size: 0.8rem;
            margin-right: 0.35rem;
            margin-bottom: 0.35rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# Classification report helpers
# =========================================================
def _parse_classification_report_text(report_text: str) -> pd.DataFrame:
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


def _report_to_dataframe(report_value: Any) -> pd.DataFrame:
    if isinstance(report_value, dict):
        try:
            return pd.DataFrame(report_value).T.reset_index().rename(columns={"index": "Клас / середнє"})
        except Exception:
            return pd.DataFrame()
    if isinstance(report_value, str):
        return _parse_classification_report_text(report_value)
    return pd.DataFrame()


def _render_classification_report(report_value: Any, title: str) -> None:
    if report_value is None:
        return

    st.markdown(f"**{title}**")
    report_df = _report_to_dataframe(report_value)
    if not report_df.empty:
        st.dataframe(report_df, use_container_width=True, hide_index=True)
        return

    if isinstance(report_value, str):
        st.code(report_value, language="text")
        return

    st.caption("Не вдалося відобразити classification report у табличному вигляді.")


# =========================================================
# Visualization helpers
# =========================================================
def _to_numeric_array(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        arr = value.astype(float, copy=False).ravel()
        return arr if arr.size else None
    if isinstance(value, (list, tuple)):
        try:
            arr = np.asarray(value, dtype=float).ravel()
            return arr if arr.size else None
        except Exception:
            return None
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except Exception:
            return None
        return _to_numeric_array(parsed)
    return None


def _plot_confusion_matrix(cm: np.ndarray, class_labels: List[Any], title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(class_labels)))
    ax.set_yticks(np.arange(len(class_labels)))
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)

    thresh = cm.max() / 2 if cm.size > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(int(cm[i, j]), "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolor="k")

    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)

    ax.set_xlabel("Actual values")
    ax.set_ylabel("Predicted values")
    ax.set_title(title)
    ax.grid(True)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _render_confusion_matrix_from_result(result: Dict[str, Any], title: str, regression_mode: bool = False) -> None:
    if regression_mode:
        cm = result.get("confusion_matrix_after_discretization")
        labels = result.get("classes_after_discretization")
        if cm is None:
            cm = result.get("confusion_matrix")
        if labels is None:
            labels = result.get("classes")
    else:
        cm = result.get("confusion_matrix")
        labels = result.get("classes")

    if cm is None:
        return

    try:
        cm_np = np.asarray(cm, dtype=int)
    except Exception:
        st.caption("Не вдалося побудувати confusion matrix: некоректний формат матриці у JSON.")
        return

    if cm_np.ndim != 2 or cm_np.shape[0] != cm_np.shape[1]:
        st.caption("Не вдалося побудувати confusion matrix: очікується квадратна матриця.")
        return

    if not labels or len(labels) != cm_np.shape[0]:
        labels = list(range(cm_np.shape[0]))

    _plot_confusion_matrix(cm_np, list(labels), title)


def _render_actual_vs_predicted_from_result(result: Dict[str, Any], title: str) -> None:
    true_keys = ["y_true_cont", "actual_values", "y_true", "actual"]
    pred_keys = ["y_pred_cont", "predicted_values", "y_pred", "predicted"]

    y_true = next((_to_numeric_array(result.get(key)) for key in true_keys if key in result), None)
    y_pred = next((_to_numeric_array(result.get(key)) for key in pred_keys if key in result), None)

    if y_true is None or y_pred is None:
        st.caption(
            "Графік Actual vs Predicted недоступний для цього JSON: у файлі немає збережених масивів "
            "фактичних і прогнозованих безперервних значень."
        )
        return

    if len(y_true) != len(y_pred) or len(y_true) == 0:
        st.caption("Не вдалося побудувати Actual vs Predicted: масиви мають різну довжину або порожні.")
        return

    _plot_actual_vs_predicted(y_true, y_pred, title)


# =========================================================
# Render helpers: summary and comparison
# =========================================================
def _comparison_df(report: StrategyReport) -> pd.DataFrame:
    df = pd.DataFrame(report.comparison_table)
    if df.empty:
        return df

    metric = report.best_model.get("metric")
    if metric and metric in df.columns:
        df = df.sort_values(metric, ascending=False).reset_index(drop=True)

    columns_to_drop = ["Strategy", "Level", "Family"]
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    return df


def _cross_strategy_best_df(classification_report: StrategyReport, regression_report: StrategyReport) -> pd.DataFrame:
    rows = []
    for report in [classification_report, regression_report]:
        best = report.best_model or {}
        rows.append(
            {
                "Стратегія": report.strategy_label,
                "Найкраща модель": _model_display_name(best.get("name", "—")),
                "F1 weighted": _safe_float(best.get("value")),
            }
        )
    return pd.DataFrame(rows)


def _render_upload_section() -> List[Any]:
    uploaded_json = st.file_uploader(
        "JSON-звіти стратегій",
        type=["json"],
        accept_multiple_files=True,
        key="module8_uploaded_jsons",
        help="Завантажте два JSON-файли по одному для кожної стратегії.",
    )

    return uploaded_json or []


def _load_uploaded_reports(uploaded_files: List[Any]) -> Tuple[Optional[StrategyReport], Optional[StrategyReport]]:
    classification_report: Optional[StrategyReport] = None
    regression_report: Optional[StrategyReport] = None

    if not uploaded_files:
        return classification_report, regression_report

    if len(uploaded_files) > 2:
        st.warning("Завантажено більше двох JSON-файлів. Будуть використані лише перші коректні файли для кожної зі стратегій.")

    for uploaded_file in uploaded_files:
        payload, error = _parse_uploaded_json(uploaded_file)
        if error:
            st.error(error)
            continue

        validation_error = _validate_report_payload(payload)
        if validation_error:
            st.error(f"Файл '{_uploaded_name(uploaded_file)}': {validation_error}")
            continue

        strategy_type = payload.get("strategy_type")
        report = _build_strategy_report(payload, _uploaded_name(uploaded_file))

        if strategy_type == CLASSIFICATION_STRATEGY:
            if classification_report is not None:
                st.warning(
                    f"Виявлено кілька JSON для стратегії класифікації. Буде використано файл '{classification_report.file_name}', "
                    f"а файл '{_uploaded_name(uploaded_file)}' пропущено."
                )
                continue
            classification_report = report
        elif strategy_type == REGRESSION_STRATEGY:
            if regression_report is not None:
                st.warning(
                    f"Виявлено кілька JSON для стратегії регресії. Буде використано файл '{regression_report.file_name}', "
                    f"а файл '{_uploaded_name(uploaded_file)}' пропущено."
                )
                continue
            regression_report = report

    return classification_report, regression_report



def _render_cross_strategy_tab(classification_report: StrategyReport, regression_report: StrategyReport) -> None:
    best_df = _cross_strategy_best_df(classification_report, regression_report)
    if not best_df.empty:
        st.markdown("**Найкращі моделі стратегій**")
        st.dataframe(best_df, use_container_width=True, hide_index=True)


# =========================================================
# Render helpers: details per strategy
# =========================================================


def _classification_metric_df(result: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for label, key in [
        ("Accuracy", "Accuracy"),
        ("Balanced Accuracy", "Balanced_accuracy"),
        ("F1 weighted", "F1_weighted"),
    ]:
        if key in result:
            rows.append({"Метрика": label, "Значення": _safe_float(result.get(key))})
    return pd.DataFrame(rows)


def _regression_metric_df(result: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for label, key in [
        ("MAE", "MAE"),
        ("MSE", "MSE"),
        ("RMSE", "RMSE"),
        ("R²", "R2"),
        ("Accuracy", "Accuracy_after_discretization"),
        ("Balanced Accuracy", "Balanced_accuracy_after_discretization"),
        ("F1 weighted", "F1_weighted_after_discretization"),
    ]:
        if key in result:
            rows.append({"Метрика": label, "Значення": _safe_float(result.get(key))})
    return pd.DataFrame(rows)


def _render_single_result_block(strategy_type: str, model_name: str, result: Dict[str, Any]) -> None:
    st.markdown(f"### {_model_display_name(model_name)}")

    if strategy_type == CLASSIFICATION_STRATEGY:
        metrics_df = _classification_metric_df(result)
        
        _render_classification_report(result.get("classification_report"), "Classification report")
        if result.get("confusion_matrix") is not None:
            st.markdown("**Confusion matrix**")
            _render_confusion_matrix_from_result(
                result,
                title=f"Confusion Matrix: {_model_display_name(model_name)}",
                regression_mode=False,
            )
        return

    metrics_df = _regression_metric_df(result)
    if not metrics_df.empty:
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    st.markdown("**Візуальна діагностика регресії**")
    _render_actual_vs_predicted_from_result(
        result,
        title=f"Actual vs Predicted: {_model_display_name(model_name)}",
    )

    report_after_discretization = result.get("classification_report_after_discretization")
    if report_after_discretization is None:
        report_after_discretization = result.get("classification_report")
    _render_classification_report(
        report_after_discretization,
        "Classification report після категоризації",
    )

    has_regression_cm = (
        result.get("confusion_matrix_after_discretization") is not None
        or result.get("confusion_matrix") is not None
    )
    if has_regression_cm:
        st.markdown("**Confusion matrix після категоризації**")
        _render_confusion_matrix_from_result(
            result,
            title=f"Confusion Matrix after discretization: {_model_display_name(model_name)}",
            regression_mode=True,
        )


def _ordered_second_level_names(strategy_type: str, second_level_results: Dict[str, Dict[str, Any]]) -> List[str]:
    preferred = ["SoftVoting", "Stacking"] if strategy_type == CLASSIFICATION_STRATEGY else ["Voting", "Stacking"]
    existing = [name for name in preferred if name in second_level_results]
    remaining = [name for name in second_level_results.keys() if name not in existing]
    return existing + remaining



def _render_strategy_comparison_table(report: StrategyReport) -> None:
    df = _comparison_df(report)
    if df.empty:
        st.warning("Підсумкова comparison table у файлі порожня.")
        return
    st.markdown("**Підсумкова таблиця порівняння моделей**")
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_strategy_details_tab(report: StrategyReport) -> None:
    _render_strategy_comparison_table(report)

    st.markdown("---")
    st.markdown("## Базові моделі")
    for model_name, result in report.first_level_results.items():
        with st.container(border=True):
            _render_single_result_block(report.strategy_type, model_name, result)

    st.markdown("---")
    st.markdown("## Ансамблі другого рівня")
    ordered_names = _ordered_second_level_names(report.strategy_type, report.second_level_results)
    for model_name in ordered_names:
        with st.container(border=True):
            _render_single_result_block(report.strategy_type, model_name, report.second_level_results[model_name])


# =========================================================
# Public page
# =========================================================
def page_strategy_comparison() -> None:
    st.subheader("Порівняння стратегій прогнозування")
    st.write("Завантажте JSON-файли стратегій та отримайте порівняльний аналіз.")
    _inject_page_style()
    uploaded_files = _render_upload_section()
    classification_report, regression_report = _load_uploaded_reports(uploaded_files)

    if len(uploaded_files) < 2:
        st.info("Поки що не завантажено жодного файлу. Використайте кнопку вище для завантаження.")
        return

    if classification_report is None or regression_report is None:
        st.warning("Не вдалося отримати JSON-файли стратегій.")
        return

    tab_compare, tab_cls, tab_reg = st.tabs(
        [
            "Порівняння стратегій",
            "Класифікація",
            "Регресія",
        ]
    )

    with tab_compare:
        _render_cross_strategy_tab(classification_report, regression_report)

    with tab_cls:
        _render_strategy_details_tab(classification_report)

    with tab_reg:
        _render_strategy_details_tab(regression_report)
