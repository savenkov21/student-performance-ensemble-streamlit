from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from utils.second_level_constants import (
    CLASSIFICATION_STRATEGY,
    REGRESSION_STRATEGY,
    SOFT_VOTING_NAME,
    STACKING_NAME,
    STRATEGY_LABELS,
    VOTING_NAME,
    SecondLevelConfig,
    SecondLevelContext,
)
from utils.second_level_core import (
    default_classification_weights,
    default_regression_weights,
    discretize_g3,
    extract_proba_groups,
    extract_regression_prediction_columns,
)


def metric(value: float) -> str:
    return f"{float(value):.4f}"


def result_value(result_dict: dict[str, Any], key: str, default: float | None = None) -> float:
    aliases = {
        "Accuracy": ["Accuracy", "Accuracy_after_discretization"],
        "Balanced_accuracy": ["Balanced_accuracy", "Balanced_accuracy_after_discretization", "Balanced_accuracyter"],
        "F1_weighted": ["F1_weighted", "F1_weighted_after_discretization"],
        "R2": ["R2", "R²"],
    }

    for candidate in aliases.get(key, [key]):
        if candidate in result_dict and result_dict[candidate] is not None:
            return float(result_dict[candidate])

    if default is not None:
        return float(default)
    raise KeyError(key)


def render_section_header(title: str, subtitle: str | None = None) -> None:
    st.subheader(title)
    if subtitle:
        st.caption(subtitle)


def plot_confusion_matrix(cm, class_labels, title) -> None:
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
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_actual_vs_predicted(y_true, y_pred, title) -> None:
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


def strategy_label(strategy_type: str) -> str:
    return STRATEGY_LABELS.get(strategy_type, strategy_type)


def build_results_overview(results: dict[str, dict[str, Any]], strategy_type: str) -> pd.DataFrame:
    rows = []
    if strategy_type == CLASSIFICATION_STRATEGY:
        for name, res in results.items():
            rows.append({
                "Ensemble": name,
                "Accuracy": result_value(res, "Accuracy"),
                "Balanced Accuracy": result_value(res, "Balanced_accuracy"),
                "F1 weighted": result_value(res, "F1_weighted"),
            })
    else:
        for name, res in results.items():
            rows.append({
                "Ensemble": name,
                "MAE": float(res["MAE"]),
                "MSE": float(res["MSE"]),
                "RMSE": float(res["RMSE"]),
                "R²": result_value(res, "R2"),
                "F1 weighted": result_value(res, "F1_weighted"),
            })
    return pd.DataFrame(rows)


def render_hero(context: SecondLevelContext) -> None:
    st.title("Навчання моделей другого рівня")
    st.caption(
        "На цьому етапі метаознаки використовуються для побудови ансамблів другого рівня. "
        "Сторінка дає короткий підсумок, порівняння ансамблів і звіти по кожному підходу."
    )
    st.write(f"**Стратегія:** {strategy_label(context.strategy_type)}")
    if context.strategy_type == CLASSIFICATION_STRATEGY:
        st.write(f"**Ансамблі:** {SOFT_VOTING_NAME}, {STACKING_NAME}")
    else:
        st.write(f"**Ансамблі:** {VOTING_NAME}, {STACKING_NAME}")


def render_experiment_snapshot(context: SecondLevelContext) -> None:
    render_section_header(
        "1. Поточний стан даних",
        "Короткий огляд активної конфігурації другого рівня, яка буде використана для побудови ансамблів другого рівня.",
    )
    st.write(f"**Цільова змінна:** {context.target_col}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Навчальна метавибірка", f"{context.meta_X_train.shape[0]}")
    c2.metric("Тестова метавибірка", f"{context.meta_X_test.shape[0]}")
    c3.metric("Метаознаки", f"{context.meta_X_train.shape[1]}")

    q1_col, q3_col = st.columns(2)
    q1_col.metric("Нижній поріг категоризації: Q1", f"{context.q1:.2f}")
    q3_col.metric("Верхній поріг категоризації: Q3", f"{context.q3:.2f}")


def render_classification_weight_controls(meta_X_test: pd.DataFrame) -> list[float] | None:
    group_names = sorted(extract_proba_groups(meta_X_test).keys())
    if not group_names:
        st.info("Probability-блоки для Soft Voting будуть визначені автоматично після формування метаознак.")
        return None

    st.markdown("**Ваги для Soft Voting**")
    st.caption("Кожна вага відповідає одному probability-блоку базової моделі.")
    cols = st.columns(len(group_names))
    defaults = default_classification_weights(meta_X_test)
    weights: list[float] = []
    for idx, group_name in enumerate(group_names):
        label = group_name.replace("meta_", "")
        with cols[idx]:
            weights.append(float(st.number_input(label, min_value=0.0, max_value=10.0, value=float(defaults[idx]), step=0.05, key=f"soft_vote_weight_{idx}")))
    return weights


def render_regression_weight_controls(meta_X_test: pd.DataFrame) -> list[float] | None:
    pred_cols = extract_regression_prediction_columns(meta_X_test)
    if not pred_cols:
        st.info("Регресійні метапрогнози для Voting будуть визначені автоматично після формування метаознак.")
        return None

    st.markdown("**Ваги для Voting**")
    st.caption("За замовчуванням використовуються рівні ваги. За потреби змініть їх вручну для зваженого усереднення регресійних метапрогнозів.")
    cols = st.columns(len(pred_cols))
    defaults = default_regression_weights(meta_X_test)
    weights: list[float] = []
    for idx, col_name in enumerate(pred_cols):
        label = str(col_name).replace("meta_", "")
        with cols[idx]:
            weights.append(float(st.number_input(label, min_value=0.0, max_value=10.0, value=float(defaults[idx]), step=0.05, key=f"reg_vote_weight_{idx}")))
    return weights


def render_controls(context: SecondLevelContext) -> tuple[SecondLevelConfig, bool, bool]:
    render_section_header(
        "2. Налаштування ансамблів другого рівня",
        "Виберіть конфігурацію усереднення та виконайте навчання ансамблів другого рівня.",
    )

    classification_weights = None
    regression_weights = None

    with st.expander("Коротко про підхід", expanded=False):
        if context.strategy_type == CLASSIFICATION_STRATEGY:
            st.write("Soft Voting усереднює ймовірності класів від базових моделей, а Stacking навчає LogisticRegression на метаознаках.")
            st.write(f"**Підхід:** {SOFT_VOTING_NAME}, {STACKING_NAME}")
            st.write(f"**Probability-блоків:** {len(extract_proba_groups(context.meta_X_test))}")
        else:
            st.write("Voting усереднює регресійні метапрогнози, а Stacking використовує Ridge як метамодель другого рівня.")
            st.write(f"**Підхід:** {VOTING_NAME} + {STACKING_NAME}")
            st.write(f"**Регресійних метапрогнозів для усереднення:** {len(extract_regression_prediction_columns(context.meta_X_test))}")
            st.write("**Ваги за замовчуванням:** рівні (1.0 для кожної моделі)")

    if context.strategy_type == CLASSIFICATION_STRATEGY:
        classification_weights = render_classification_weight_controls(context.meta_X_test)
    else:
        regression_weights = render_regression_weight_controls(context.meta_X_test)

    b1, b2 = st.columns([1.6, 1])
    train_clicked = b1.button("Навчити ансамблі другого рівня", use_container_width=True, type="primary")
    clear_clicked = b2.button("Очистити результати", use_container_width=True)

    return SecondLevelConfig(classification_weights=classification_weights, regression_weights=regression_weights), train_clicked, clear_clicked


def render_second_level_classification_result(context: SecondLevelContext, ensemble_name, result_dict, y_pred_cls):
    st.markdown(f"#### {ensemble_name}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", metric(result_value(result_dict, "Accuracy")))
    c2.metric("Balanced Accuracy", metric(result_value(result_dict, "Balanced_accuracy")))
    c3.metric("F1 weighted", metric(result_value(result_dict, "F1_weighted")))

    tab1, tab2, tab3 = st.tabs(["Звіт", "Confusion matrix", "Preview"])
    with tab1:
        st.dataframe(pd.DataFrame(result_dict["classification_report"]).T, use_container_width=True)
    with tab2:
        plot_confusion_matrix(np.asarray(result_dict["confusion_matrix"]), result_dict["classes"], f"Confusion Matrix: {ensemble_name}")
    with tab3:
        preview_df = pd.DataFrame({"y_true_cls": np.asarray(context.y_test_cls), "y_pred_cls": np.asarray(y_pred_cls)})
        st.dataframe(preview_df.head(20), use_container_width=True)
        if ensemble_name == SOFT_VOTING_NAME:
            avg_proba = st.session_state.get("second_level_predictions", {}).get(SOFT_VOTING_NAME)
            if avg_proba is not None:
                avg_proba = np.asarray(avg_proba)
                st.markdown("**Усереднені ймовірності**")
                st.dataframe(pd.DataFrame(avg_proba, columns=[f"avg_proba_class_{i}" for i in range(avg_proba.shape[1])]).head(20), use_container_width=True)
            used_groups = result_dict.get("used_probability_groups")
            if used_groups:
                st.caption("Використані probability-блоки: " + ", ".join(str(group).replace("meta_", "") for group in used_groups))


def render_second_level_regression_result(context: SecondLevelContext, ensemble_name, result_dict, y_pred_cont, y_pred_cls):
    st.markdown(f"#### {ensemble_name}")
    tab1, tab2, tab3 = st.tabs(["Regression Evaluation", "Classification Evaluation After Categorization", "Preview"])
    with tab1:
        reg_df = pd.DataFrame({
            "Metric": ["MAE", "MSE", "RMSE", "R²"],
            "Value": [
                metric(result_dict["MAE"]),
                metric(result_dict["MSE"]),
                metric(result_dict["RMSE"]),
                metric(result_value(result_dict, "R2")),
            ],
        })
        st.dataframe(reg_df, hide_index=True, use_container_width=True)
        plot_actual_vs_predicted(np.asarray(context.y_test_cont), np.asarray(y_pred_cont), f"Actual vs Predicted: {ensemble_name}")
    with tab2:
        st.caption("Класифікаційні метрики для регресійної стратегії обчислюються після категоризації безперервних прогнозів за порогами Q1 і Q3.")
        cat_df = pd.DataFrame({
            "Metric": ["Accuracy", "Balanced accuracy", "F1 weighted"],
            "Value": [
                metric(result_value(result_dict, "Accuracy")),
                metric(result_value(result_dict, "Balanced_accuracy")),
                metric(result_value(result_dict, "F1_weighted")),
            ],
        })
        st.dataframe(cat_df, hide_index=True, use_container_width=True)
        st.dataframe(pd.DataFrame(result_dict["classification_report"]).T, use_container_width=True)
        plot_confusion_matrix(np.asarray(result_dict["confusion_matrix"]), result_dict["classes"], f"Confusion Matrix for regression: {ensemble_name}")
    with tab3:
        preview_df = pd.DataFrame({
            "y_true_cont": np.asarray(context.y_test_cont),
            "y_pred_cont": np.asarray(y_pred_cont),
            "y_true_cls": np.asarray(discretize_g3(context.y_test_cont, context.q1, context.q3)),
            "y_pred_cls": np.asarray(y_pred_cls),
        })
        st.dataframe(preview_df.head(20), use_container_width=True)


def render_results(context: SecondLevelContext) -> None:
    if "second_level_results" not in st.session_state:
        return
    if st.session_state.get("second_level_strategy_type") != context.strategy_type:
        st.warning("Стратегія змінилася. Перенавчіть ансамблі другого рівня.")
        return

    results = st.session_state.second_level_results
    render_section_header("3. Порівняння результатів", "Підсумок і таблиця метрик для ансамблів другого рівня.")
    overview_df = build_results_overview(results, context.strategy_type).rename(columns={"Ensemble": "Ансамбль"})
    st.dataframe(overview_df, hide_index=True, use_container_width=True)

    st.caption("Таблиця має порівняльний характер. Автоматичний вибір ансамблю не виконується.")

    if context.strategy_type == CLASSIFICATION_STRATEGY:
        tab1, tab2 = st.tabs(["Soft Voting", "Stacking"])
        with tab1:
            render_second_level_classification_result(context, SOFT_VOTING_NAME, results[SOFT_VOTING_NAME], st.session_state.second_level_predictions_cls[SOFT_VOTING_NAME])
        with tab2:
            render_second_level_classification_result(context, STACKING_NAME, results[STACKING_NAME], st.session_state.second_level_predictions_cls[STACKING_NAME])
    else:
        tab1, tab2 = st.tabs(["Voting", "Stacking"])
        with tab1:
            render_second_level_regression_result(context, VOTING_NAME, results[VOTING_NAME], st.session_state.second_level_predictions[VOTING_NAME], st.session_state.second_level_predictions_cls[VOTING_NAME])
        with tab2:
            render_second_level_regression_result(context, STACKING_NAME, results[STACKING_NAME], st.session_state.second_level_predictions[STACKING_NAME], st.session_state.second_level_predictions_cls[STACKING_NAME])
