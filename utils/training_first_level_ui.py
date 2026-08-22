from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from utils.training_first_level_constants import (
    CLASSIFICATION_STRATEGY,
    MODEL_DISPLAY_NAMES,
    OptimizationConfig,
    REGRESSION_STRATEGY,
    STRATEGY_LABELS,
    TrainingContext,
)


def strategy_label(strategy_type: str) -> str:
    return STRATEGY_LABELS.get(strategy_type, strategy_type)


def cv_metric_label(strategy_type: str) -> str:
    if strategy_type == CLASSIFICATION_STRATEGY:
        return "Середній CV F1-macro"
    return "Середній CV score (neg. MSE)"


def model_display_name(model_name: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model_name, model_name)


def metric(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def params_to_dataframe(params_dict: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for key, value in params_dict.items():
        normalized = "None" if value is None else round(value, 6) if isinstance(value, float) else value
        rows.append({"Параметр": key, "Значення": normalized})
    return pd.DataFrame(rows)


def result_metric(result: dict[str, Any], *keys: str, default: float | None = None) -> float:
    for key in keys:
        if key in result and result[key] is not None:
            return float(result[key])
    if default is not None:
        return float(default)
    raise KeyError(keys[0])


def result_report(result: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("classification_report", "classification_report_after_discretization"):
        if result.get(key) is not None:
            return result[key]
    return None


def result_confusion_payload(result: dict[str, Any]) -> tuple[np.ndarray, list[Any]] | tuple[None, None]:
    if result.get("confusion_matrix") is not None:
        return np.asarray(result["confusion_matrix"]), result.get("classes")
    if result.get("confusion_matrix_after_discretization") is not None:
        return np.asarray(result["confusion_matrix_after_discretization"]), result.get("classes_after_discretization")
    return None, None


def classification_metric_bundle(result: dict[str, Any]) -> dict[str, float]:
    return {
        "Accuracy": result_metric(result, "Accuracy", "Accuracy_after_discretization"),
        "Balanced Accuracy": result_metric(
            result,
            "Balanced_accuracy",
            "Balanced_accuracy_after_discretization",
            "Balanced_accuracyter",
        ),
        "F1 weighted": result_metric(result, "F1_weighted", "F1_weighted_after_discretization"),
    }


def build_comparison_dataframe(results: dict[str, dict[str, Any]], strategy_type: str) -> pd.DataFrame:
    rows: dict[str, dict[str, float]] = {}
    for model_name, model_res in results.items():
        cls_metrics = classification_metric_bundle(model_res)
        if strategy_type == CLASSIFICATION_STRATEGY:
            rows[model_name] = cls_metrics
        else:
            rows[model_name] = {
                "MAE": float(model_res["MAE"]),
                "MSE": float(model_res["MSE"]),
                "RMSE": float(model_res["RMSE"]),
                "R2": float(model_res["R2"]),
                **cls_metrics,
            }
    return pd.DataFrame(rows).T


def section_header(title: str, subtitle: str | None = None) -> None:
    st.subheader(title)
    if subtitle:
        st.caption(subtitle)


def plot_confusion_matrix(cm: np.ndarray, class_labels: list[Any], model_name: str, title_suffix: str = "") -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    title = f"Confusion Matrix: {model_display_name(model_name)}"
    if title_suffix:
        title += f" {title_suffix}"
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


def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolor="k")
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)
    ax.set_xlabel("Actual values")
    ax.set_ylabel("Predicted values")
    ax.set_title(f"Actual vs Predicted: {model_display_name(model_name)}")
    ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_page_intro(context: TrainingContext) -> None:
    st.markdown("### Навчання базових моделей першого рівня")
    st.write(
        "На цьому етапі виконується Bayesian Optimization для базових моделей, а далі — їх навчання на підготовлених даних."
    )
    st.caption("Моделі: Random Forest, XGBoost, Ridge.")


def render_data_snapshot(context: TrainingContext) -> None:
    section_header(
        "1. Поточний стан даних",
        "Огляд конфігурації, яка буде використана для оптимізації та навчання моделей.",
    )
    st.write(f"**Стратегія:** {strategy_label(context.strategy_type)}")
    st.write(f"**Цільова змінна:** {context.target_col}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Навчальна вибірка", context.X_train.shape[0])
    c2.metric("Тестова вибірка", context.X_test.shape[0])
    c3.metric("Ознаки", context.X_train.shape[1])

    with st.expander("Показати деталі розбиття та категоризації", expanded=False):
        d1, d2, d3 = st.columns(3)
        d1.metric("Q1", metric(context.q1))
        d2.metric("Q3", metric(context.q3))
        d3.metric("Ознак після preprocessing", str(context.X_train.shape[1]))

        if context.strategy_type == CLASSIFICATION_STRATEGY:
            st.info(
                "Для прямої багатокласової класифікації неперервна цільова змінна категоризується за порогами Q1 і Q3, обчисленими лише на train-вибірці."
            )
        else:
            st.info(
                "Для регресійної стратегії моделі прогнозують неперервне значення, а класифікаційні метрики додатково обчислюються після категоризації прогнозів."
            )

        st.dataframe(
            pd.DataFrame({"Ознака": list(context.X_train.columns)}),
            use_container_width=True,
            hide_index=True,
        )


def render_bayesian_block(strategy_type: str) -> tuple[OptimizationConfig, bool, bool]:
    section_header(
        "2. Оптимізація гіперпараметрів: Bayesian Optimization",
        "Підбір найкращих гіперпараметрів для Ridge, Random Forest і XGBoost.",
    )

    left, right = st.columns([1.15, 0.85])
    with left:
        with st.container(border=True):
            st.markdown("**Налаштування оптимізації**")
            c1, c2 = st.columns(2)
            with c1:
                init_points = st.slider(
                    "init_points",
                    min_value=3,
                    max_value=20,
                    value=int(st.session_state.get("bo_init_points", 10)),
                    step=1,
                    help="Кількість стартових випадкових конфігурацій.",
                )
            with c2:
                n_iter = st.slider(
                    "n_iter",
                    min_value=5,
                    max_value=40,
                    value=int(st.session_state.get("bo_n_iter", 30)),
                    step=1,
                    help="Кількість основних ітерацій пошуку.",
                )

            st.session_state.bo_init_points = init_points
            st.session_state.bo_n_iter = n_iter
            st.caption(f"Метрика оптимізації: {cv_metric_label(strategy_type)}")

            b1, b2 = st.columns([1.6, 1])
            optimize_clicked = b1.button("Запустити оптимізацію", use_container_width=True, type="primary")
            clear_clicked = b2.button("Очистити", use_container_width=True)

    with right:
        with st.container(border=True):
            st.markdown("**Коротко про метод**")
            st.write(
                "Bayesian Optimization не перебирає всі комбінації параметрів, а послідовно вибирає найперспективніші конфігурації на основі попередніх оцінок."
            )
            st.write(
                "Для класифікації оптимізується F1-macro у CV, а для регресійної стратегії — CV score на основі neg. MSE."
            )

    return OptimizationConfig(init_points=init_points, n_iter=n_iter), optimize_clicked, clear_clicked


def render_optimization_results(optimization_results: dict[str, Any] | None, strategy_type: str) -> None:
    section_header(
        "3. Результати оптимізації",
        "Оптимальні конфігурації, які будуть використані для навчання моделей першого рівня.",
    )
    if not optimization_results:
        st.info("Після запуску оптимізації тут з'являться найкращі конфігурації для кожної моделі.")
        return

    summary_rows = []
    metric_col = cv_metric_label(strategy_type)
    for model_name, result in optimization_results.items():
        summary_rows.append(
            {
                "Модель": model_display_name(model_name),
                metric_col: round(float(result["best_cv_score"]), 4),
                "К-сть параметрів": len(result.get("best_params", {})),
            }
        )
    summary_df = pd.DataFrame(summary_rows)

    best_idx = summary_df[metric_col].idxmax()
    best_score = summary_df.loc[best_idx, metric_col]

    c1, c2 = st.columns(2)
    c1.metric("Оптимізовано моделей", str(len(summary_df)))
    c2.metric("Найкращий CV результат", metric(best_score))

    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    tabs = st.tabs([model_display_name(name) for name in optimization_results.keys()])
    for tab, (model_name, result) in zip(tabs, optimization_results.items()):
        with tab:
            left, right = st.columns([0.8, 1.2])
            left.metric(metric_col, metric(result["best_cv_score"]))
            right.dataframe(
                params_to_dataframe(result.get("best_params", {})),
                use_container_width=True,
                hide_index=True,
            )


def render_training_controls() -> tuple[bool, bool]:
    section_header(
        "4. Навчання базових моделей",
        "Після оптимізації гіперпараметрів виконайте навчання Ridge, Random Forest і XGBoost на train-вибірці.",
    )
    with st.container(border=True):
        c1, c2 = st.columns([1.6, 1])
        train_clicked = c1.button(
            "Навчити моделі першого рівня",
            use_container_width=True,
            type="primary",
        )
        clear_clicked = c2.button(
            "Очистити результати навчання",
            use_container_width=True,
        )
    return train_clicked, clear_clicked

def render_results_summary(results: dict[str, dict[str, Any]], strategy_type: str) -> pd.DataFrame:
    section_header("5. Порівняння результатів")
    comparison_df = build_comparison_dataframe(results, strategy_type)
    comparison_df.index = [model_display_name(name) for name in comparison_df.index]

    display_df = comparison_df.reset_index().rename(columns={"index": "Модель", "R2": "R²"})
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    if strategy_type == REGRESSION_STRATEGY:
        st.caption(
            "Для регресійної стратегії Accuracy, Balanced Accuracy та F1 weighted обчислено після категоризації неперервних прогнозів за порогами Q1 і Q3."
        )

    return comparison_df


def render_classification_model_details(selected_model: str, selected_result: dict[str, Any]) -> None:
    metrics = classification_metric_bundle(selected_result)
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", metric(metrics["Accuracy"]))
    c2.metric("Balanced Accuracy", metric(metrics["Balanced Accuracy"]))
    c3.metric("F1 weighted", metric(metrics["F1 weighted"]))

    report = result_report(selected_result)
    if report is not None:
        st.dataframe(pd.DataFrame(report).T, use_container_width=True)

    cm, labels = result_confusion_payload(selected_result)
    if cm is not None and labels is not None:
        plot_confusion_matrix(cm, labels, selected_model)


def render_regression_model_details(
    selected_model: str,
    selected_result: dict[str, Any],
    predictions: dict[str, np.ndarray],
    y_test_cont: pd.Series,
) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE", metric(selected_result["MAE"]))
    c2.metric("MSE", metric(selected_result["MSE"]))
    c3.metric("RMSE", metric(selected_result["RMSE"]))
    c4.metric("R²", metric(selected_result["R2"]))

    plot_actual_vs_predicted(
        np.asarray(y_test_cont),
        np.asarray(predictions[selected_model]),
        selected_model,
    )

    cls_metrics = classification_metric_bundle(selected_result)
    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy", metric(cls_metrics["Accuracy"]))
    m2.metric("Balanced Accuracy", metric(cls_metrics["Balanced Accuracy"]))
    m3.metric("F1 weighted", metric(cls_metrics["F1 weighted"]))
    st.caption("Ці класифікаційні метрики для регресійної моделі обчислено після категоризації прогнозів.")

    report = result_report(selected_result)
    if report is not None:
        st.dataframe(pd.DataFrame(report).T, use_container_width=True)

    cm, labels = result_confusion_payload(selected_result)
    if cm is not None and labels is not None:
        plot_confusion_matrix(cm, labels, selected_model, title_suffix="(після категоризації)")


def render_model_details_tabs(
    results: dict[str, dict[str, Any]],
    strategy_type: str,
    predictions: dict[str, np.ndarray],
    y_test_cont: pd.Series,
) -> None:

    tabs = st.tabs([model_display_name(name) for name in results.keys()])
    for tab, model_name in zip(tabs, results.keys()):
        with tab:
            st.markdown(f"**{model_display_name(model_name)}**")
            selected_result = results[model_name]
            if strategy_type == CLASSIFICATION_STRATEGY:
                render_classification_model_details(model_name, selected_result)
            else:
                render_regression_model_details(model_name, selected_result, predictions, y_test_cont)


def render_results_if_ready(context: TrainingContext) -> None:
    results = st.session_state.get("first_level_results")
    if results is None:
        st.info("Після навчання тут з’явиться порівняння моделей та детальні метрики першого рівня.")
        return
    if st.session_state.get("first_level_strategy_type") != context.strategy_type:
        st.warning("Стратегія змінилася. Перенавчіть базові моделі.")
        return

    render_results_summary(results, context.strategy_type)
    render_model_details_tabs(
        results=results,
        strategy_type=context.strategy_type,
        predictions=st.session_state.get("first_level_predictions", {}),
        y_test_cont=context.y_test_cont,
    )
