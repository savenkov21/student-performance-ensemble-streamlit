from __future__ import annotations

import hashlib

import streamlit as st

from utils.strategy_comparison_prediction import render_student_rating_prediction_tab
from utils.strategy_comparison_ui import (
    load_uploaded_reports,
    load_uploaded_statistical_report,
    render_cross_strategy_tab,
    render_statistical_analysis_report_tab,
    render_statistical_report_upload,
    render_strategy_details_tab,
    render_upload_section,
)


def page_strategy_comparison() -> None:
    st.subheader("Порівняння стратегій прогнозування")
    st.write("Завантажте JSON-файли стратегій та отримайте порівняльний аналіз.")

    uploaded_files = render_upload_section()
    classification_report, regression_report = load_uploaded_reports(uploaded_files)

    source_signature = hashlib.sha256()
    for uploaded_file in uploaded_files:
        source_signature.update(uploaded_file.name.encode("utf-8"))
        source_signature.update(uploaded_file.getvalue())
    comparison_signature = source_signature.hexdigest()
    reports_ready = classification_report is not None and regression_report is not None

    if st.button(
        "Виконати порівняння стратегій",
        type="primary",
        use_container_width=True,
        disabled=not reports_ready,
        help=None if reports_ready else "Завантажте по одному коректному JSON-звіту для кожної з двох стратегій.",
    ):
        st.session_state["module9_comparison_signature"] = comparison_signature
    comparison_requested = (
        reports_ready
        and st.session_state.get("module9_comparison_signature") == comparison_signature
    )

    st.markdown("---")
    uploaded_statistical_report = render_statistical_report_upload()
    statistical_report = load_uploaded_statistical_report(uploaded_statistical_report)

    tab_compare, tab_cls, tab_reg, tab_statistics, tab_predict = st.tabs(
        [
            "Порівняння стратегій",
            "Класифікація",
            "Регресія",
            "Статистичний аналіз",
            "Прогноз рейтингу студента",
        ]
    )

    with tab_compare:
        if len(uploaded_files) < 2:
            st.info("Для міжстратегічного порівняння завантажте два JSON-файли окремо для класифікації та регресії.")
        elif classification_report is None or regression_report is None:
            st.warning("Не вдалося отримати коректні JSON-файли стратегій.")
        elif not comparison_requested:
            st.info("JSON-звіти перевірено. Натисніть «Виконати порівняння стратегій».")
        else:
            render_cross_strategy_tab(classification_report, regression_report)

    with tab_cls:
        if classification_report is None:
            st.info("Завантажте JSON-звіт стратегії класифікації, щоб переглянути детальні результати.")
        else:
            render_strategy_details_tab(classification_report)

    with tab_reg:
        if regression_report is None:
            st.info("Завантажте JSON-звіт стратегії регресії, щоб переглянути детальні результати.")
        else:
            render_strategy_details_tab(regression_report)

    with tab_statistics:
        if statistical_report is None:
            st.info("Завантажте JSON-звіт статистичного аналізу, сформований у попередньому модулі.")
        else:
            render_statistical_analysis_report_tab(statistical_report)

    with tab_predict:
        render_student_rating_prediction_tab()
