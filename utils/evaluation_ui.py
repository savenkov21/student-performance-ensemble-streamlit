from __future__ import annotations

from typing import Any

import streamlit as st
import pandas as pd

from utils.evaluation_constants import CLASSIFICATION_STRATEGY, EvaluationArtifacts, EvaluationContext
from utils.evaluation_core import (
    build_export_filename,
    classification_metric_rows,
    get_primary_metric,
    get_result_value,
    metric_value,
    ordered_second_level_names,
    prepare_metric_comparison_df,
    results_to_json_bytes,
    strategy_label,
)
from utils.evaluation_reports import render_classification_report


def render_classification_metrics(result: dict[str, Any]) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", metric_value(result["Accuracy"]))
    c2.metric("Balanced Accuracy", metric_value(result["Balanced_accuracy"]))
    c3.metric("F1 weighted", metric_value(result["F1_weighted"]))


def render_regression_metrics(result: dict[str, Any]) -> None:
    st.markdown("**Регресійні метрики**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE", metric_value(get_result_value(result, "MAE", default=0.0)))
    c2.metric("MSE", metric_value(get_result_value(result, "MSE", default=0.0)))
    c3.metric("RMSE", metric_value(get_result_value(result, "RMSE", default=0.0)))
    c4.metric("R²", metric_value(get_result_value(result, "R2", "R²", default=0.0)))

    st.markdown("**Після категоризації**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", metric_value(get_result_value(result, "Accuracy", "Accuracy_after_discretization", default=0.0)))
    c2.metric(
        "Balanced Accuracy",
        metric_value(
            get_result_value(
                result,
                "Balanced_accuracy",
                "Balanced_accuracy_after_discretization",
                "Balanced_accuracyter",
                default=0.0,
            )
        ),
    )
    c3.metric(
        "F1 weighted",
        metric_value(get_result_value(result, "F1_weighted", "F1_weighted_after_discretization", default=0.0)),
    )


def render_result_block(strategy_type: str, title: str, result: dict[str, Any]) -> None:
    st.markdown(f"### {title}")

    if strategy_type == CLASSIFICATION_STRATEGY:
        render_classification_metrics(result)
        st.dataframe(classification_metric_rows(result), use_container_width=True, hide_index=True)

        used_probability_groups = result.get("used_probability_groups")
        if used_probability_groups:
            used_groups_text = ", ".join(str(group).replace("meta_", "") for group in used_probability_groups)
            st.caption(f"Використані probability-блоки: {used_groups_text}")

        if "classification_report" in result:
            render_classification_report(result["classification_report"], "Classification report")
        return

    render_regression_metrics(result)
    report_value = get_result_value(
        result,
        "classification_report",
        "classification_report_after_discretization",
        default=None,
    )
    if report_value is not None:
        render_classification_report(report_value, "Звіт після категоризації")


def render_page_header(context: EvaluationContext) -> None:
    st.title("Оцінювання якості моделей")
    st.caption("Підсумковий етап зі зведеним порівнянням базових моделей та ансамблів другого рівня.")
    st.markdown(f"**Стратегія:** {strategy_label(context.strategy_type)}")


def render_experiment_snapshot(context: EvaluationContext) -> None:
    st.subheader("1. Поточний стан експерименту")
    st.markdown(f"**Цільова змінна:** {context.target_col}")

    c1, c2 = st.columns(2)
    c1.metric("Нижній поріг категоризації: Q1", metric_value(context.q1))
    c2.metric("Верхній поріг категоризації: Q3", metric_value(context.q3))


def render_comparison_section(context: EvaluationContext, artifacts: EvaluationArtifacts) -> None:
    metric = get_primary_metric(context.strategy_type)
    st.subheader("2. Порівняння результатів")
    st.caption("Зведений підсумок для моделей першого рівня та ансамблів другого рівня.")

    metric_df = prepare_metric_comparison_df(artifacts.comparison_df, metric)

    st.markdown(f"#### Результати моделей за {metric}")
    summary_columns = [column for column in ["Model", "Level", "Family", metric] if column in metric_df.columns]
    summary_df = metric_df[summary_columns].copy()
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    with st.expander("Показати повну таблицю метрик", expanded=False):
        st.dataframe(artifacts.comparison_df.copy(), use_container_width=True, hide_index=True)


def render_details_section(context: EvaluationContext) -> None:
    st.subheader("3. Деталі моделей")
    st.caption("Розгорнуті метрики для базових моделей і ансамблів другого рівня.")

    second_level_names = ordered_second_level_names(context.strategy_type, context.second_level_results)
    if len(second_level_names) >= 2:
        tab_titles = ["Базові моделі", second_level_names[0], second_level_names[1]]
    else:
        tab_titles = ["Базові моделі", "Ансамблі 2-го рівня"]

    tabs = st.tabs(tab_titles)

    with tabs[0]:
        for model_name, result in context.first_level_results.items():
            with st.container(border=True):
                render_result_block(context.strategy_type, model_name, result)

    if len(second_level_names) >= 2:
        with tabs[1]:
            with st.container(border=True):
                render_result_block(
                    context.strategy_type,
                    second_level_names[0],
                    context.second_level_results[second_level_names[0]],
                )
        with tabs[2]:
            with st.container(border=True):
                render_result_block(
                    context.strategy_type,
                    second_level_names[1],
                    context.second_level_results[second_level_names[1]],
                )
        if len(second_level_names) > 2:
            with st.expander("Додаткові ансамблі другого рівня"):
                for extra_name in second_level_names[2:]:
                    with st.container(border=True):
                        render_result_block(
                            context.strategy_type,
                            extra_name,
                            context.second_level_results[extra_name],
                        )
        return

    with tabs[1]:
        for ensemble_name, result in context.second_level_results.items():
            with st.container(border=True):
                render_result_block(context.strategy_type, ensemble_name, result)


def render_export_section(context: EvaluationContext, artifacts: EvaluationArtifacts) -> None:
    st.subheader("4. Експорт підсумкового звіту")
    st.caption("Збереження підсумкового JSON-звіту для поточної стратегії з параметрами та метриками всіх моделей.")

    json_bytes = results_to_json_bytes(artifacts.final_report)
    st.download_button(
        label="Завантажити підсумковий звіт (JSON)",
        data=json_bytes,
        file_name=build_export_filename(context.strategy_type),
        mime="application/json",
        use_container_width=True,
    )
    st.info(
        "Назва файлу автоматично містить назву стратегії та дату завантаження, "
        "щоб JSON-файли було легко розрізняти."
    )
