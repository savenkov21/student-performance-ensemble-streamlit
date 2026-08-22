from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from utils.strategy_comparison_constants import CLASSIFICATION_STRATEGY, REGRESSION_STRATEGY, StrategyReport
from utils.strategy_comparison_core import (
    classification_metric_df,
    comparison_df,
    cross_strategy_models_df,
    model_display_name,
    ordered_second_level_names,
    regression_metric_df,
)
from utils.strategy_comparison_reports import (
    build_strategy_report,
    parse_uploaded_json,
    render_classification_report,
    uploaded_name,
    validate_report_payload,
)
from utils.statistical_analysis_reports import validate_statistical_report
from utils.statistical_analysis_ui import render_saved_statistical_report
from utils.strategy_comparison_visuals import (
    render_actual_vs_predicted_from_result,
    render_confusion_matrix_from_result,
)


def render_upload_section() -> List[Any]:
    uploaded_json = st.file_uploader(
        "JSON-звіти стратегій",
        type=["json"],
        accept_multiple_files=True,
        key="module9_uploaded_jsons",
        help="Завантажте два JSON-файли по одному для кожної стратегії.",
    )
    return uploaded_json or []



def render_statistical_report_upload() -> Any:
    return st.file_uploader(
        "JSON-звіт статистичного аналізу",
        type=["json"],
        accept_multiple_files=False,
        key="module9_statistical_analysis_json",
        help="Завантажте JSON, сформований на сторінці 'Статистичний аналіз результатів класифікації'.",
    )


def load_uploaded_statistical_report(uploaded_file: Any) -> Optional[Dict[str, Any]]:
    if uploaded_file is None:
        return None

    payload, error = parse_uploaded_json(uploaded_file)
    if error:
        st.error(error)
        return None
    assert payload is not None

    validation_error = validate_statistical_report(payload)
    if validation_error:
        st.error(f"Файл '{uploaded_name(uploaded_file)}': {validation_error}")
        return None
    return payload


def render_statistical_analysis_report_tab(report: Dict[str, Any]) -> None:
    st.title("Статистичний аналіз результатів класифікації")
    render_saved_statistical_report(report)


def load_uploaded_reports(uploaded_files: List[Any]) -> Tuple[Optional[StrategyReport], Optional[StrategyReport]]:
    classification_report: Optional[StrategyReport] = None
    regression_report: Optional[StrategyReport] = None

    if not uploaded_files:
        return classification_report, regression_report

    if len(uploaded_files) > 2:
        st.warning("Завантажено більше двох JSON-файлів. Будуть використані лише перші коректні файли для кожної зі стратегій.")

    for uploaded_file in uploaded_files:
        payload, error = parse_uploaded_json(uploaded_file)
        if error:
            st.error(error)
            continue

        validation_error = validate_report_payload(payload)
        if validation_error:
            st.error(f"Файл '{uploaded_name(uploaded_file)}': {validation_error}")
            continue

        strategy_type = payload.get("strategy_type")
        report = build_strategy_report(payload, uploaded_name(uploaded_file))

        if strategy_type == CLASSIFICATION_STRATEGY:
            if classification_report is not None:
                st.warning(
                    f"Виявлено кілька JSON для стратегії класифікації. Буде використано файл '{classification_report.file_name}', "
                    f"а файл '{uploaded_name(uploaded_file)}' пропущено."
                )
                continue
            classification_report = report
        elif strategy_type == REGRESSION_STRATEGY:
            if regression_report is not None:
                st.warning(
                    f"Виявлено кілька JSON для стратегії регресії. Буде використано файл '{regression_report.file_name}', "
                    f"а файл '{uploaded_name(uploaded_file)}' пропущено."
                )
                continue
            regression_report = report

    return classification_report, regression_report


def render_cross_strategy_tab(classification_report: StrategyReport, regression_report: StrategyReport) -> None:
    models_df = cross_strategy_models_df(classification_report, regression_report)
    if models_df.empty:
        st.warning("У завантажених звітах відсутні дані для міжстратегічного порівняння.")
        return

    st.markdown("**Результати моделей двох стратегій**")
    st.dataframe(models_df, use_container_width=True, hide_index=True)
    st.caption(
        "Таблиця відображає результати всіх моделей. "
    )


def render_strategy_comparison_table(report: StrategyReport) -> None:
    df = comparison_df(report)
    if df.empty:
        st.warning("Підсумкова comparison table у файлі порожня.")
        return
    st.markdown("**Підсумкова таблиця порівняння моделей**")
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_single_result_block(strategy_type: str, model_name: str, result: Dict[str, Any]) -> None:
    st.markdown(f"### {model_display_name(model_name)}")

    if strategy_type == CLASSIFICATION_STRATEGY:
        metrics_df = classification_metric_df(result)
        if not metrics_df.empty:
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        render_classification_report(result.get("classification_report"), "Classification report")
        if result.get("confusion_matrix") is not None:
            st.markdown("**Confusion matrix**")
            render_confusion_matrix_from_result(
                result,
                title=f"Confusion Matrix: {model_display_name(model_name)}",
                regression_mode=False,
            )
        return

    metrics_df = regression_metric_df(result)
    if not metrics_df.empty:
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    st.markdown("**Візуальна діагностика регресії**")
    render_actual_vs_predicted_from_result(
        result,
        title=f"Actual vs Predicted: {model_display_name(model_name)}",
    )

    report_after_discretization = result.get("classification_report_after_discretization")
    if report_after_discretization is None:
        report_after_discretization = result.get("classification_report")
    render_classification_report(
        report_after_discretization,
        "Classification report після категоризації",
    )

    has_regression_cm = (
        result.get("confusion_matrix_after_discretization") is not None
        or result.get("confusion_matrix") is not None
    )
    if has_regression_cm:
        st.markdown("**Confusion matrix після категоризації**")
        render_confusion_matrix_from_result(
            result,
            title=f"Confusion Matrix after discretization: {model_display_name(model_name)}",
            regression_mode=True,
        )


def render_strategy_details_tab(report: StrategyReport) -> None:
    render_strategy_comparison_table(report)

    st.markdown("---")
    st.markdown("## Базові моделі")
    for model_name, result in report.first_level_results.items():
        with st.container(border=True):
            render_single_result_block(report.strategy_type, model_name, result)

    st.markdown("---")
    st.markdown("## Ансамблі другого рівня")
    ordered_names = ordered_second_level_names(report.strategy_type, report.second_level_results)
    for model_name in ordered_names:
        with st.container(border=True):
            render_single_result_block(report.strategy_type, model_name, report.second_level_results[model_name])
