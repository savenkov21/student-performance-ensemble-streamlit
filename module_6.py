from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


CLASSIFICATION_STRATEGY = "direct_multiclass_classification"
REGRESSION_STRATEGY = "regression_with_categorization"
RANDOM_STATE = 42
SOFT_VOTING_NAME = "SoftVoting"
VOTING_NAME = "Voting"
STACKING_NAME = "Stacking"


@dataclass
class SecondLevelContext:
    meta_X_train: pd.DataFrame
    meta_X_test: pd.DataFrame
    strategy_type: str
    q1: float
    q3: float
    target_col: str | None
    y_train_cls: pd.Series | None = None
    y_test_cls: pd.Series | None = None
    y_train_cont: pd.Series | None = None
    y_test_cont: pd.Series | None = None


@dataclass
class SecondLevelConfig:
    classification_weights: list[float] | None
    regression_weights: list[float] | None


@dataclass
class SecondLevelArtifacts:
    models: dict[str, Any]
    results: dict[str, dict[str, Any]]
    predictions: dict[str, Any]
    predictions_cls: dict[str, np.ndarray]
    params: dict[str, dict[str, Any]]


# =========================================================
# Common helpers
# =========================================================
def discretize_g3(y, q1, q3):
    y = np.asarray(y)
    cls = np.zeros_like(y, dtype=int)
    cls[y <= q1] = 0
    cls[(y > q1) & (y <= q3)] = 1
    cls[y > q3] = 2
    return cls


def _check_ready_for_second_level() -> bool:
    required_keys = [
        "meta_X_train",
        "meta_X_test",
        "strategy_type",
        "meta_ready",
        "meta_strategy_type",
        "Q1",
        "Q3",
    ]
    missing = [k for k in required_keys if k not in st.session_state]

    if missing:
        st.warning("Спочатку завантажте дані на сторінці 'Завантаження та первинний аналіз даних'.")
        return False

    if not st.session_state.get("meta_ready", False):
        st.warning("Спочатку виконайте формування метаознак на сторінці 'Формування метаознак'.")
        return False

    if st.session_state.get("meta_strategy_type") != st.session_state.get("strategy_type"):
        st.warning("Стратегія змінилася. Сформуйте метаознаки повторно.")
        return False

    return True


def _clear_second_level_state() -> None:
    keys = [
        "second_level_models",
        "second_level_results",
        "second_level_predictions",
        "second_level_predictions_cls",
        "second_level_params",
        "second_level_strategy_type",
    ]
    for key in keys:
        st.session_state.pop(key, None)


def _safe_classification_report(y_true, y_pred) -> dict[str, Any]:
    return classification_report(y_true, y_pred, output_dict=True, zero_division=0)


def _metric(value: float) -> str:
    return f"{float(value):.4f}"


def _result_value(result_dict: dict[str, Any], key: str, default: float | None = None) -> float:
    aliases = {
        "Accuracy": ["Accuracy", "Accuracy_after_discretization"],
        "Balanced_accuracy": [
            "Balanced_accuracy",
            "Balanced_accuracy_after_discretization",
            "Balanced_accuracyter",
        ],
        "F1_weighted": ["F1_weighted", "F1_weighted_after_discretization"],
        "R2": ["R2", "R²"],
    }

    for candidate in aliases.get(key, [key]):
        if candidate in result_dict and result_dict[candidate] is not None:
            return float(result_dict[candidate])

    if default is not None:
        return float(default)

    raise KeyError(key)


def _inject_page_style() -> None:
    st.markdown(
        """
        <style>
        .m4-hero {
            padding: 1rem 1.15rem;
            border: 1px solid rgba(120,120,120,0.20);
            border-radius: 14px;
            background: linear-gradient(180deg, rgba(70,130,180,0.08), rgba(255,255,255,0.02));
            margin-bottom: 0.8rem;
        }
        .m4-section-title {
            margin-top: 0.5rem;
            margin-bottom: 0.15rem;
            font-weight: 700;
            font-size: 1.05rem;
        }
        .m4-subtle {
            color: rgba(120,120,120,0.95);
            font-size: 0.95rem;
            margin-bottom: 0.2rem;
        }
        .m4-card {
            border: 1px solid rgba(120,120,120,0.18);
            border-radius: 14px;
            padding: 0.8rem 0.95rem;
            background: rgba(255,255,255,0.02);
            min-height: 110px;
        }
        .m4-card-kicker {
            font-size: 0.82rem;
            color: rgba(120,120,120,0.95);
            margin-bottom: 0.35rem;
        }
        .m4-card-value {
            font-size: 1.2rem;
            font-weight: 700;
            line-height: 1.3;
            margin-bottom: 0.3rem;
        }
        .m4-card-note {
            font-size: 0.86rem;
            color: rgba(120,120,120,0.92);
        }
        .m4-pill {
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


def _render_section_header(title: str, subtitle: str | None = None) -> None:
    st.markdown(f'<div class="m4-section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="m4-subtle">{subtitle}</div>', unsafe_allow_html=True)


def _plot_confusion_matrix(cm, class_labels, title) -> None:
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
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _plot_actual_vs_predicted(y_true, y_pred, title) -> None:
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

# =========================================================
# Metric builders
# =========================================================
def _classification_metrics_dict(model_name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1_weighted = f1_score(y_true, y_pred, average="weighted")

    report = _safe_classification_report(y_true, y_pred)
    labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "model_name": model_name,
        "Accuracy": float(acc),
        "Balanced_accuracy": float(bal_acc),
        "F1_weighted": float(f1_weighted),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "classes": labels.tolist(),
    }


def _regression_with_discretization_metrics_dict(model_name, y_true_cont, y_pred_cont, q1, q3):
    mae = mean_absolute_error(y_true_cont, y_pred_cont)
    mse = mean_squared_error(y_true_cont, y_pred_cont)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true_cont, y_pred_cont)

    y_true_cls = discretize_g3(y_true_cont, q1, q3)
    y_pred_cls = discretize_g3(y_pred_cont, q1, q3)

    acc_cls = accuracy_score(y_true_cls, y_pred_cls)
    bal_acc_cls = balanced_accuracy_score(y_true_cls, y_pred_cls)
    f1_weighted_cls = f1_score(y_true_cls, y_pred_cls, average="weighted")

    report_cls = _safe_classification_report(y_true_cls, y_pred_cls)
    labels_cls = np.unique(np.concatenate([np.asarray(y_true_cls), np.asarray(y_pred_cls)]))
    cm_cls = confusion_matrix(y_true_cls, y_pred_cls, labels=labels_cls)

    return {
        "model_name": model_name,
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2": float(r2),
        "Accuracy": float(acc_cls),
        "Balanced_accuracy": float(bal_acc_cls),
        "F1_weighted": float(f1_weighted_cls),
        "classification_report": report_cls,
        "confusion_matrix": cm_cls.tolist(),
        "classes": labels_cls.tolist(),
        "y_true_cont": np.asarray(y_true_cont, dtype=float).tolist(),
        "y_pred_cont": np.asarray(y_pred_cont, dtype=float).tolist(),
    }


# =========================================================
# Classification helpers
# =========================================================
def _extract_proba_groups(meta_X) -> dict[str, dict[int, str]]:
    meta_df = pd.DataFrame(meta_X).copy()
    proba_groups: dict[str, dict[int, str]] = {}

    for col in meta_df.columns:
        if "_proba_" not in col:
            continue

        model_part, class_part = col.rsplit("_proba_", 1)
        try:
            class_idx = int(class_part)
        except ValueError:
            continue

        proba_groups.setdefault(model_part, {})
        proba_groups[model_part][class_idx] = col

    return proba_groups


def _soft_voting_from_meta_proba(meta_X, classes, weights=None):
    meta_df = pd.DataFrame(meta_X).copy()
    classes = list(classes)

    proba_groups = _extract_proba_groups(meta_df)
    if not proba_groups:
        raise ValueError(
            "Для Soft Voting не знайдено probability-колонок у meta_X_test. "
            "Переконайтеся, що сфоормовано класифікаційні метаознаки через predict_proba."
        )

    ordered_group_names = sorted(proba_groups.keys())
    model_prob_arrays = []

    for group_name in ordered_group_names:
        class_map = proba_groups[group_name]
        missing_classes = [cls for cls in classes if int(cls) not in class_map]
        if missing_classes:
            raise ValueError(f"У probability-блоці {group_name} відсутні класи: {missing_classes}")

        ordered_cols = [class_map[int(cls)] for cls in classes]
        model_prob_arrays.append(meta_df[ordered_cols].to_numpy(dtype=float))

    stacked = np.stack(model_prob_arrays, axis=0)

    if weights is None:
        avg_proba = np.mean(stacked, axis=0)
    else:
        weights = np.asarray(weights, dtype=float)
        if len(weights) != len(model_prob_arrays):
            raise ValueError(
                f"Кількість ваг ({len(weights)}) не збігається з кількістю базових моделей "
                f"({len(model_prob_arrays)})."
            )
        if np.isclose(weights.sum(), 0.0):
            raise ValueError("Сума ваг для Soft Voting не може дорівнювати 0.")
        avg_proba = np.average(stacked, axis=0, weights=weights)

    y_pred_cls = np.asarray(classes)[np.argmax(avg_proba, axis=1)]
    return avg_proba, y_pred_cls, ordered_group_names


def _extract_regression_prediction_columns(meta_X) -> list[str]:
    meta_df = pd.DataFrame(meta_X).copy()
    return [
        col
        for col in meta_df.columns
        if str(col).startswith("meta_") and "_proba_" not in str(col)
    ]


def _default_regression_weights(meta_X_test: pd.DataFrame) -> list[float]:
    pred_cols = _extract_regression_prediction_columns(meta_X_test)
    return [1.0] * len(pred_cols)


# =========================================================
# Context / config
# =========================================================
def _load_context() -> SecondLevelContext:
    strategy_type = st.session_state.strategy_type
    context = SecondLevelContext(
        meta_X_train=st.session_state.meta_X_train,
        meta_X_test=st.session_state.meta_X_test,
        strategy_type=strategy_type,
        q1=float(st.session_state.Q1),
        q3=float(st.session_state.Q3),
        target_col=st.session_state.get("target_col"),
    )

    if strategy_type == CLASSIFICATION_STRATEGY:
        context.y_train_cls = pd.Series(st.session_state.y_train).copy()
        context.y_test_cls = pd.Series(st.session_state.y_test).copy()
    else:
        context.y_train_cont = pd.Series(st.session_state.y_train_cont).copy()
        context.y_test_cont = pd.Series(st.session_state.y_test_cont).copy()

    return context


def _strategy_label(strategy_type: str) -> str:
    mapping = {
        CLASSIFICATION_STRATEGY: "Пряма багатокласова класифікація",
        REGRESSION_STRATEGY: "Регресія з подальшою категоризацією",
    }
    return mapping.get(strategy_type, strategy_type)


def _primary_metric_name(strategy_type: str) -> str:
    return "F1_weighted"



def _build_results_overview(results: dict[str, dict[str, Any]], strategy_type: str) -> pd.DataFrame:
    rows = []
    if strategy_type == CLASSIFICATION_STRATEGY:
        for name, res in results.items():
            rows.append(
                {
                    "Ensemble": name,
                    "Accuracy": _result_value(res, "Accuracy"),
                    "Balanced Accuracy": _result_value(res, "Balanced_accuracy"),
                    "F1 weighted": _result_value(res, "F1_weighted"),
                }
            )
    else:
        for name, res in results.items():
            rows.append(
                {
                    "Ensemble": name,
                    "MAE": float(res["MAE"]),
                    "MSE": float(res["MSE"]),
                    "RMSE": float(res["RMSE"]),
                    "R²": _result_value(res, "R2"),
                    "F1 weighted": _result_value(res, "F1_weighted"),
                }
            )
    return pd.DataFrame(rows)


def _best_ensemble_name(results: dict[str, dict[str, Any]], strategy_type: str) -> str:
    metric = _primary_metric_name(strategy_type)
    return max(results.items(), key=lambda kv: _result_value(kv[1], metric))[0]



def _render_hero(context: SecondLevelContext) -> None:
    strategy = _strategy_label(context.strategy_type)
    st.markdown(
        f'''
        <div class="m4-hero">
            <div style="font-size:1.2rem;font-weight:700;margin-bottom:0.3rem;">Навчання моделей другого рівня</div>
            <div style="margin-bottom:0.6rem;">На цьому етапі метаознаки використовуються для побудови фінального ансамблю. Сторінка дає короткий підсумок, порівняння ансамблів і звіти по кожному підходу.</div>
            <span class="m4-pill">{SOFT_VOTING_NAME if context.strategy_type == CLASSIFICATION_STRATEGY else VOTING_NAME}</span>
            <span class="m4-pill">{STACKING_NAME}</span>
        </div>
        ''',
        unsafe_allow_html=True,
    )


def _render_experiment_snapshot(context: SecondLevelContext) -> None:
    _render_section_header(
        "1. Поточний стан даних",
        "Короткий огляд активної конфігурації другого рівня, яка буде використана для побудови фінального ансамблю.",
    )

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
    st.markdown(f"**Стратегія:** {_strategy_label(context.strategy_type)}")
    st.markdown(f"**Цільова змінна:** {context.target_col}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Навчальна метавибірка", f"{context.meta_X_train.shape[0]}")
    c2.metric("Тестова метавибірка", f"{context.meta_X_test.shape[0]}")
    c3.metric("Ознаки", f"{context.meta_X_train.shape[1]}")
    
    c1, c2 = st.columns(2)
    c1.metric("Нижній поріг категоризації: Q1", f"{context.q1:.0f}")
    c2.metric("Верхній поріг категоризації: Q3", f"{context.q3:.0f}")
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)


def _default_classification_weights(meta_X_test: pd.DataFrame) -> list[float]:
    group_count = len(_extract_proba_groups(meta_X_test))
    if group_count <= 0:
        return []
    return [1.0] * group_count


def _render_classification_weight_controls(meta_X_test: pd.DataFrame) -> list[float] | None:
    group_names = sorted(_extract_proba_groups(meta_X_test).keys())
    if not group_names:
        st.info("Probability-блоки для Soft Voting будуть визначені автоматично після формування метаознак.")
        return None

    st.markdown("**Ваги для Soft Voting**")
    st.caption("Кожна вага відповідає одному probability-блоку базової моделі.")
    cols = st.columns(len(group_names))
    defaults = _default_classification_weights(meta_X_test)
    weights: list[float] = []

    for idx, group_name in enumerate(group_names):
        label = group_name.replace("meta_", "")
        with cols[idx]:
            weights.append(
                float(
                    st.number_input(
                        label,
                        min_value=0.0,
                        max_value=10.0,
                        value=float(defaults[idx]),
                        step=0.05,
                        key=f"soft_vote_weight_{idx}",
                    )
                )
            )
    return weights


def _render_regression_weight_controls(meta_X_test: pd.DataFrame) -> list[float] | None:
    pred_cols = _extract_regression_prediction_columns(meta_X_test)
    if not pred_cols:
        st.info("Регресійні метапрогнози для Voting будуть визначені автоматично після формування метаознак.")
        return None

    st.markdown("**Ваги для Voting**")
    st.caption("За замовчуванням використовуються рівні ваги. За потреби змініть їх вручну для зваженого усереднення регресійних метапрогнозів.")
    cols = st.columns(len(pred_cols))
    defaults = _default_regression_weights(meta_X_test)
    weights: list[float] = []

    for idx, col_name in enumerate(pred_cols):
        label = str(col_name).replace("meta_", "")
        with cols[idx]:
            weights.append(
                float(
                    st.number_input(
                        label,
                        min_value=0.0,
                        max_value=10.0,
                        value=float(defaults[idx]),
                        step=0.05,
                        key=f"reg_vote_weight_{idx}",
                    )
                )
            )
    return weights



def _render_controls(context: SecondLevelContext) -> tuple[SecondLevelConfig, bool, bool]:
    _render_section_header(
        "2. Налаштування ансамблів другого рівня",
        "Виберіть конфігурацію усереднення та виконайте навчання фінальних ансамблів.",
    )

    classification_weights = None
    regression_weights = None

    with st.expander("Коротко про підхід", expanded=False):
        if context.strategy_type == CLASSIFICATION_STRATEGY:
            st.write(
                    "Soft Voting усереднює ймовірності класів від базових моделей, а Stacking навчає LogisticRegression на метаознаках."
                )
            
            st.write(f"**Підхід:** {SOFT_VOTING_NAME}, {STACKING_NAME}")
            st.write(f"**Probability-блоків:** {len(_extract_proba_groups(context.meta_X_test))}")
        else:
            st.write(
                    "Voting усереднює регресійні метапрогнози, а Stacking використовує Ridge як метамодель другого рівня."
                )
            with st.expander("Поточна конфігурація", expanded=True):
                st.write(f"**Підхід:** {VOTING_NAME} + {STACKING_NAME}")
                st.write(f"**Регресійних метапрогнозів для усереднення:** {len(_extract_regression_prediction_columns(context.meta_X_test))}")
                st.write("**Ваги за замовчуванням:** рівні (1.0 для кожної моделі)")
    
    with st.container(border=True):
        if context.strategy_type == CLASSIFICATION_STRATEGY:
            classification_weights = _render_classification_weight_controls(context.meta_X_test)
        else:
            regression_weights = _render_regression_weight_controls(context.meta_X_test)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        b1, b2 = st.columns([1.6, 1])
        train_clicked = b1.button("Навчити ансамблі другого рівня", use_container_width=True, type="primary")
        clear_clicked = b2.button("Очистити результати", use_container_width=True)

    return SecondLevelConfig(
        classification_weights=classification_weights,
        regression_weights=regression_weights,
    ), train_clicked, clear_clicked

def _train_soft_voting_classifier(meta_X_test, y_test_cls, classes, weights=None):
    avg_proba, y_pred_cls, used_groups = _soft_voting_from_meta_proba(meta_X=meta_X_test, classes=classes, weights=weights)
    results = _classification_metrics_dict(model_name=SOFT_VOTING_NAME, y_true=y_test_cls, y_pred=y_pred_cls)
    results["used_probability_groups"] = used_groups
    return avg_proba, np.asarray(y_pred_cls), results


def _train_stacking_classifier(meta_X_train, meta_X_test, y_train_cls, y_test_cls):
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(meta_X_train, y_train_cls)
    y_pred_cls = model.predict(meta_X_test)
    results = _classification_metrics_dict(model_name=STACKING_NAME, y_true=y_test_cls, y_pred=y_pred_cls)
    return model, np.asarray(y_pred_cls), results


def _train_voting_regressor_from_meta(meta_X_test, y_test_cont, q1, q3, weights=None):
    meta_df = pd.DataFrame(meta_X_test).copy()
    pred_cols = _extract_regression_prediction_columns(meta_df)
    if not pred_cols:
        raise ValueError("Для Voting Regressor не знайдено регресійних метапрогнозів у meta_X_test.")

    meta_np = meta_df[pred_cols].to_numpy(dtype=float)

    if weights is None:
        weights = np.ones(len(pred_cols), dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
        if len(weights) != len(pred_cols):
            raise ValueError(
                f"Кількість ваг ({len(weights)}) не збігається з кількістю регресійних метапрогнозів ({len(pred_cols)})."
            )
        if np.isclose(weights.sum(), 0.0):
            raise ValueError("Сума ваг для Voting Regressor не може дорівнювати 0.")

    y_pred_cont = np.average(meta_np, axis=1, weights=weights)

    y_pred_cls = discretize_g3(y_pred_cont, q1, q3)
    results = _regression_with_discretization_metrics_dict(
        model_name=VOTING_NAME,
        y_true_cont=y_test_cont,
        y_pred_cont=y_pred_cont,
        q1=q1,
        q3=q3,
    )
    results["used_regression_prediction_columns"] = pred_cols
    results["used_weights"] = weights.astype(float).tolist()
    return np.asarray(y_pred_cont), np.asarray(y_pred_cls), results


def _train_stacking_regressor(meta_X_train, meta_X_test, y_train_cont, y_test_cont, q1, q3):
    model = Ridge(alpha=1.0)
    model.fit(meta_X_train, y_train_cont)
    y_pred_cont = model.predict(meta_X_test)
    y_pred_cls = discretize_g3(y_pred_cont, q1, q3)
    results = _regression_with_discretization_metrics_dict(
        model_name=STACKING_NAME,
        y_true_cont=y_test_cont,
        y_pred_cont=y_pred_cont,
        q1=q1,
        q3=q3,
    )
    return model, np.asarray(y_pred_cont), np.asarray(y_pred_cls), results


def _train_classification_ensembles(context: SecondLevelContext, config: SecondLevelConfig) -> SecondLevelArtifacts:
    assert context.y_train_cls is not None and context.y_test_cls is not None
    classes = np.unique(np.asarray(context.y_train_cls))

    vote_avg_proba, vote_pred_cls, vote_results = _train_soft_voting_classifier(
        meta_X_test=context.meta_X_test,
        y_test_cls=context.y_test_cls,
        classes=classes,
        weights=config.classification_weights,
    )
    stack_model, stack_pred_cls, stack_results = _train_stacking_classifier(
        meta_X_train=context.meta_X_train,
        meta_X_test=context.meta_X_test,
        y_train_cls=context.y_train_cls,
        y_test_cls=context.y_test_cls,
    )

    return SecondLevelArtifacts(
        models={SOFT_VOTING_NAME: None, STACKING_NAME: stack_model},
        results={SOFT_VOTING_NAME: vote_results, STACKING_NAME: stack_results},
        predictions={SOFT_VOTING_NAME: vote_avg_proba, STACKING_NAME: None},
        predictions_cls={SOFT_VOTING_NAME: vote_pred_cls, STACKING_NAME: stack_pred_cls},
        params={
            SOFT_VOTING_NAME: {
                "weights": config.classification_weights,
                "aggregation": "weighted mean of probabilities",
            },
            STACKING_NAME: {
                "model_name": "LogisticRegression",
                "max_iter": 1000,
                "random_state": RANDOM_STATE,
            },
        },
    )


def _train_regression_ensembles(context: SecondLevelContext, config: SecondLevelConfig) -> SecondLevelArtifacts:
    assert context.y_train_cont is not None and context.y_test_cont is not None

    vote_pred_cont, vote_pred_cls, vote_results = _train_voting_regressor_from_meta(
        meta_X_test=context.meta_X_test,
        y_test_cont=context.y_test_cont,
        q1=context.q1,
        q3=context.q3,
        weights=config.regression_weights,
    )
    stack_model, stack_pred_cont, stack_pred_cls, stack_results = _train_stacking_regressor(
        meta_X_train=context.meta_X_train,
        meta_X_test=context.meta_X_test,
        y_train_cont=context.y_train_cont,
        y_test_cont=context.y_test_cont,
        q1=context.q1,
        q3=context.q3,
    )

    return SecondLevelArtifacts(
        models={VOTING_NAME: None, STACKING_NAME: stack_model},
        results={VOTING_NAME: vote_results, STACKING_NAME: stack_results},
        predictions={VOTING_NAME: vote_pred_cont, STACKING_NAME: stack_pred_cont},
        predictions_cls={VOTING_NAME: vote_pred_cls, STACKING_NAME: stack_pred_cls},
        params={
            VOTING_NAME: {
                "weights": config.regression_weights,
                "aggregation": "weighted mean of regression predictions",
                "used_prediction_columns": vote_results.get("used_regression_prediction_columns", []),
            },
            STACKING_NAME: {
                "model_name": "Ridge",
                "alpha": 1.0,
            },
        },
    )


def _run_training(context: SecondLevelContext, config: SecondLevelConfig) -> SecondLevelArtifacts:
    if context.strategy_type == CLASSIFICATION_STRATEGY:
        return _train_classification_ensembles(context, config)
    return _train_regression_ensembles(context, config)


def _save_artifacts(context: SecondLevelContext, artifacts: SecondLevelArtifacts) -> None:
    st.session_state.second_level_models = artifacts.models
    st.session_state.second_level_results = artifacts.results
    st.session_state.second_level_predictions = artifacts.predictions
    st.session_state.second_level_predictions_cls = artifacts.predictions_cls
    st.session_state.second_level_params = artifacts.params
    st.session_state.second_level_strategy_type = context.strategy_type


# =========================================================
# Rendering results
# =========================================================
def _render_second_level_classification_result(context: SecondLevelContext, ensemble_name, result_dict, y_pred_cls):
    st.markdown(f"#### {ensemble_name}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", _metric(_result_value(result_dict, "Accuracy")))
    c2.metric("Balanced Accuracy", _metric(_result_value(result_dict, "Balanced_accuracy")))
    c3.metric("F1 weighted", _metric(_result_value(result_dict, "F1_weighted")))

    tab1, tab2, tab3 = st.tabs(["Звіт", "Confusion matrix", "Preview"])

    with tab1:
        report_df = pd.DataFrame(result_dict["classification_report"]).T
        st.dataframe(report_df, use_container_width=True)

    with tab2:
        cm = np.asarray(result_dict["confusion_matrix"])
        class_labels = result_dict["classes"]
        _plot_confusion_matrix(cm, class_labels, f"Confusion Matrix: {ensemble_name}")

    with tab3:
        preview_df = pd.DataFrame(
            {
                "y_true_cls": np.asarray(context.y_test_cls),
                "y_pred_cls": np.asarray(y_pred_cls),
            }
        )
        st.dataframe(preview_df.head(20), use_container_width=True)

        if ensemble_name == SOFT_VOTING_NAME:
            avg_proba = st.session_state.get("second_level_predictions", {}).get(SOFT_VOTING_NAME)
            if avg_proba is not None:
                avg_proba = np.asarray(avg_proba)
                proba_cols = [f"avg_proba_class_{i}" for i in range(avg_proba.shape[1])]
                proba_df = pd.DataFrame(avg_proba, columns=proba_cols)
                st.markdown("**Усереднені ймовірності**")
                st.dataframe(proba_df.head(20), use_container_width=True)

            used_groups = result_dict.get("used_probability_groups")
            if used_groups:
                used_groups_text = ", ".join(str(group).replace("meta_", "") for group in used_groups)
                st.caption(f"Використані probability-блоки: {used_groups_text}")


def _render_second_level_regression_result(context: SecondLevelContext, ensemble_name, result_dict, y_pred_cont, y_pred_cls):
    st.markdown(f"#### {ensemble_name}")

    tab1, tab2, tab3 = st.tabs(["Regression Evaluation", "Classification Evaluation After Categorization", "Preview"])

    with tab1:
        reg_df = pd.DataFrame(
            {
                "Metric": ["MAE", "MSE", "RMSE", "R²"],
                "Value": [
                    _metric(result_dict["MAE"]),
                    _metric(result_dict["MSE"]),
                    _metric(result_dict["RMSE"]),
                    _metric(_result_value(result_dict, "R2")),
                ],
            }
        )
        st.dataframe(reg_df, hide_index=True, use_container_width=True)
        _plot_actual_vs_predicted(np.asarray(context.y_test_cont), np.asarray(y_pred_cont), f"Actual vs Predicted: {ensemble_name}")

    with tab2:
        st.caption("Класифікаційні метрики для регресійної стратегії обчислюються після категоризації безперервних прогнозів за порогами Q1 і Q3.")
        cat_df = pd.DataFrame(
            {
                "Metric": ["Accuracy", "Balanced accuracy", "F1 weighted"],
                "Value": [
                    _metric(_result_value(result_dict, "Accuracy")),
                    _metric(_result_value(result_dict, "Balanced_accuracy")),
                    _metric(_result_value(result_dict, "F1_weighted")),
                ],
            }
        )
        st.dataframe(cat_df, hide_index=True, use_container_width=True)
        report_df = pd.DataFrame(result_dict["classification_report"]).T
        st.dataframe(report_df, use_container_width=True)
        cm = np.asarray(result_dict["confusion_matrix"])
        class_labels = result_dict["classes"]
        _plot_confusion_matrix(cm, class_labels, f"Confusion Matrix for regression: {ensemble_name}")

    with tab3:
        preview_df = pd.DataFrame(
            {
                "y_true_cont": np.asarray(context.y_test_cont),
                "y_pred_cont": np.asarray(y_pred_cont),
                "y_true_cls": np.asarray(discretize_g3(context.y_test_cont, context.q1, context.q3)),
                "y_pred_cls": np.asarray(y_pred_cls),
            }
        )
        st.dataframe(preview_df.head(20), use_container_width=True)


def _render_results(context: SecondLevelContext) -> None:
    if "second_level_results" not in st.session_state:
        return

    if st.session_state.get("second_level_strategy_type") != context.strategy_type:
        st.warning("Стратегія змінилася. Перенавчіть ансамблі другого рівня.")
        return

    results = st.session_state.second_level_results
    

    _render_section_header(
        "3. Порівняння результатів",
        "Підсумок і таблиця метрик для фінальних ансамблів другого рівня.",
    )
    
    overview_df = _build_results_overview(results, context.strategy_type)
    display_df = overview_df.rename(columns={"Ensemble": "Ансамбль"})
    st.dataframe(
        display_df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Ансамбль": st.column_config.TextColumn("Ансамбль", width="large"),
        },
    )
    
    best_name = _best_ensemble_name(results, context.strategy_type)
    metric_name = _primary_metric_name(context.strategy_type)
    best_metric_value = _result_value(results[best_name], metric_name)
    st.success(f"Найкращий ансамбль другого рівня є **{best_name}** за головною метрикою '{metric_name}': {_metric(best_metric_value)}.")

    if context.strategy_type == CLASSIFICATION_STRATEGY:
        tab1, tab2 = st.tabs(["Soft Voting", "Stacking"])
        with tab1:
            _render_second_level_classification_result(
                context=context,
                ensemble_name=SOFT_VOTING_NAME,
                result_dict=results[SOFT_VOTING_NAME],
                y_pred_cls=st.session_state.second_level_predictions_cls[SOFT_VOTING_NAME],
            )
        with tab2:
            _render_second_level_classification_result(
                context=context,
                ensemble_name=STACKING_NAME,
                result_dict=results[STACKING_NAME],
                y_pred_cls=st.session_state.second_level_predictions_cls[STACKING_NAME],
            )
    else:
        tab1, tab2 = st.tabs(["Voting", "Stacking"])
        with tab1:
            _render_second_level_regression_result(
                context=context,
                ensemble_name=VOTING_NAME,
                result_dict=results[VOTING_NAME],
                y_pred_cont=st.session_state.second_level_predictions[VOTING_NAME],
                y_pred_cls=st.session_state.second_level_predictions_cls[VOTING_NAME],
            )
        with tab2:
            _render_second_level_regression_result(
                context=context,
                ensemble_name=STACKING_NAME,
                result_dict=results[STACKING_NAME],
                y_pred_cont=st.session_state.second_level_predictions[STACKING_NAME],
                y_pred_cls=st.session_state.second_level_predictions_cls[STACKING_NAME],
            )


def page_training_second_level_model():
    _inject_page_style()
    if not _check_ready_for_second_level():
        return

    context = _load_context()
    _render_hero(context)
    _render_experiment_snapshot(context)
    config, train_clicked, clear_clicked = _render_controls(context)

    if clear_clicked:
        _clear_second_level_state()
        st.success("Результати другого рівня очищено.")

    if train_clicked:
        try:
            with st.spinner("Триває навчання ансамблів другого рівня..."):
                artifacts = _run_training(context, config)
                _save_artifacts(context, artifacts)
            st.success("Ансамблі другого рівня успішно навчені.")
        except ValueError as exc:
            st.error(f"Помилка під час навчання: {exc}")

    _render_results(context)