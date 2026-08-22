from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from utils.strategy_comparison_constants import (
    CLASSIFICATION_STRATEGY,
    PREDICTION_REQUIRED_KEYS,
    REGRESSION_STRATEGY,
    LivePredictionContext,
)
from utils.strategy_comparison_core import metric_value, model_display_name, strategy_label


def session_ready_for_prediction() -> tuple[bool, list[str]]:
    missing = [key for key in PREDICTION_REQUIRED_KEYS if key not in st.session_state]
    if missing:
        return False, missing
    if not st.session_state.get("split_ready", False):
        return False, ["split_ready"]
    if not st.session_state.get("meta_ready", False):
        return False, ["meta_ready"]
    return True, []


def build_live_prediction_context() -> LivePredictionContext:
    return LivePredictionContext(
        strategy_type=st.session_state.strategy_type,
        preprocessor=st.session_state.preprocessor,
        feature_cols=list(st.session_state.get("feature_cols", [])),
        x_raw=pd.DataFrame(st.session_state.get("X_raw")),
        first_level_models=st.session_state.get("first_level_models", {}),
        first_level_results=st.session_state.get("first_level_results", {}),
        second_level_models=st.session_state.get("second_level_models", {}),
        second_level_results=st.session_state.get("second_level_results", {}),
        second_level_params=st.session_state.get("second_level_params", {}),
        meta_full_train_models=st.session_state.get("meta_full_train_models", {}),
        meta_generation_params=st.session_state.get("meta_generation_params", {}),
        q1=float(st.session_state.get("Q1")),
        q3=float(st.session_state.get("Q3")),
        feature_names_after_preprocessing=st.session_state.get("feature_names_after_preprocessing"),
    )


def available_prediction_models(context: LivePredictionContext) -> list[str]:
    ordered: list[str] = []
    for name in list(context.first_level_models.keys()) + list(context.second_level_results.keys()):
        if name not in ordered:
            ordered.append(name)
    return ordered


def render_prediction_model_selector(context: LivePredictionContext) -> str | None:
    available_models = available_prediction_models(context)
    if not available_models:
        st.warning("Немає доступних навчених моделей для прогнозування.")
        return None

    selected = st.selectbox(
        "Модель для практичного прогнозування",
        options=[None, *available_models],
        index=0,
        format_func=lambda value: "— Оберіть модель —" if value is None else model_display_name(value),
        key="module9_prediction_model_select",
        help="Модель обирає користувач. Автоматичний вибір за тестовими метриками не виконується.",
    )

    if selected is None:
        st.info("Оберіть модель, яку потрібно використати для прогнозування нового студента.")
        return None

    st.markdown(f"**Поточна стратегія:** {strategy_label(context.strategy_type)}")
    st.markdown(f"**Обрана модель:** {model_display_name(selected)}")
    return str(selected)


def default_value_for_feature(df: pd.DataFrame, column: str) -> Any:
    series = df[column] if column in df.columns else pd.Series(dtype=float)
    non_na = series.dropna()
    if non_na.empty:
        return 0.0 if pd.api.types.is_numeric_dtype(series) else ""
    if pd.api.types.is_numeric_dtype(series):
        return float(non_na.median())
    mode = non_na.mode(dropna=True)
    if not mode.empty:
        return mode.iloc[0]
    return non_na.iloc[0]


def render_student_input_form(feature_cols: list[str], x_raw: pd.DataFrame) -> pd.DataFrame:
    st.markdown("**Введення ознак нового студента**")
    defaults = {col: default_value_for_feature(x_raw, col) for col in feature_cols}
    values: dict[str, Any] = {}

    with st.form("student_rating_prediction_form"):
        cols = st.columns(2)
        for idx, feature in enumerate(feature_cols):
            container = cols[idx % 2]
            series = x_raw[feature] if feature in x_raw.columns else pd.Series(dtype=float)
            default = defaults[feature]

            with container:
                if pd.api.types.is_numeric_dtype(series):
                    clean = series.dropna()
                    min_val = float(clean.min()) if not clean.empty else 0.0
                    max_val = float(clean.max()) if not clean.empty else max(1.0, float(default))
                    step = 1.0 if pd.api.types.is_integer_dtype(series) else 0.1
                    values[feature] = st.number_input(
                        feature,
                        value=float(default),
                        step=step,
                        min_value=min_val,
                        max_value=max_val,
                        key=f"predict_input_{feature}",
                    )
                else:
                    options = [str(x) for x in pd.Series(series.dropna().astype(str).unique()).tolist()]
                    if not options:
                        options = [str(default)] if str(default) else [""]
                    default_str = str(default)
                    if default_str not in options:
                        options = [default_str] + options
                    default_index = options.index(default_str) if default_str in options else 0
                    values[feature] = st.selectbox(
                        feature,
                        options=options,
                        index=default_index,
                        key=f"predict_input_{feature}",
                    )

        submitted = st.form_submit_button("Спрогнозувати рейтинг", use_container_width=True, type="primary")

    if not submitted:
        return pd.DataFrame()
    return pd.DataFrame([values], columns=feature_cols)


def transform_student_row(context: LivePredictionContext, raw_df: pd.DataFrame) -> pd.DataFrame:
    transformed = context.preprocessor.transform(raw_df)
    columns = context.feature_names_after_preprocessing
    return pd.DataFrame(transformed, columns=columns, index=raw_df.index)


def build_live_meta_features(context: LivePredictionContext, x_prepared: pd.DataFrame) -> pd.DataFrame:
    passthrough = bool(context.meta_generation_params.get("passthrough", False))

    blocks = []
    feature_names: list[str] = []
    for model_name, model in context.meta_full_train_models.items():
        if context.strategy_type == CLASSIFICATION_STRATEGY:
            proba = np.asarray(model.predict_proba(x_prepared), dtype=float)
            class_labels = getattr(model, "classes_", np.arange(proba.shape[1]))
            block_names = [f"meta_{model_name}_proba_{cls}" for cls in class_labels]
            blocks.append(pd.DataFrame(proba, columns=block_names, index=x_prepared.index))
            feature_names.extend(block_names)
        else:
            pred = np.asarray(model.predict(x_prepared), dtype=float).reshape(-1, 1)
            block_names = [f"meta_{model_name}"]
            blocks.append(pd.DataFrame(pred, columns=block_names, index=x_prepared.index))
            feature_names.extend(block_names)

    meta_df = pd.concat(blocks, axis=1) if blocks else pd.DataFrame(index=x_prepared.index)
    if passthrough:
        meta_df = pd.concat([x_prepared.reset_index(drop=True), meta_df.reset_index(drop=True)], axis=1)
        meta_df.columns = list(x_prepared.columns) + feature_names
        meta_df.index = x_prepared.index
    return meta_df


def classification_from_model(context: LivePredictionContext, model_name: str, x_prepared: pd.DataFrame, meta_features: pd.DataFrame) -> dict[str, Any]:
    if model_name in context.first_level_models:
        model = context.first_level_models[model_name]
        pred_cls = np.asarray(model.predict(x_prepared)).ravel()
        proba = np.asarray(model.predict_proba(x_prepared), dtype=float) if hasattr(model, "predict_proba") else None
        return {"predicted_class": int(pred_cls[0]), "probabilities": proba[0].tolist() if proba is not None else None}

    if model_name == "SoftVoting":
        weights = context.second_level_params.get("SoftVoting", {}).get("weights")
        full_models = list(context.meta_full_train_models.values())
        probas = [np.asarray(model.predict_proba(x_prepared), dtype=float)[0] for model in full_models]
        probas_np = np.vstack(probas)
        if weights is None:
            weights = np.ones(probas_np.shape[0], dtype=float)
        weights = np.asarray(weights, dtype=float)
        avg_proba = np.average(probas_np, axis=0, weights=weights)
        pred_cls = int(np.argmax(avg_proba))
        return {"predicted_class": pred_cls, "probabilities": avg_proba.tolist()}

    if model_name == "Stacking":
        model = context.second_level_models.get("Stacking")
        pred_cls = np.asarray(model.predict(meta_features)).ravel()
        proba = np.asarray(model.predict_proba(meta_features), dtype=float) if hasattr(model, "predict_proba") else None
        return {"predicted_class": int(pred_cls[0]), "probabilities": proba[0].tolist() if proba is not None else None}

    raise ValueError(f"Непідтримувана модель для класифікації: {model_name}")


def regression_from_model(context: LivePredictionContext, model_name: str, x_prepared: pd.DataFrame, meta_features: pd.DataFrame) -> dict[str, Any]:
    if model_name in context.first_level_models:
        model = context.first_level_models[model_name]
        pred_cont = float(np.asarray(model.predict(x_prepared)).ravel()[0])
    elif model_name == "Voting":
        weights = context.second_level_params.get("Voting", {}).get("weights")
        preds = [float(np.asarray(model.predict(x_prepared)).ravel()[0]) for model in context.meta_full_train_models.values()]
        preds_np = np.asarray(preds, dtype=float)
        if weights is None:
            weights = np.ones(preds_np.shape[0], dtype=float)
        weights = np.asarray(weights, dtype=float)
        pred_cont = float(np.average(preds_np, weights=weights))
    elif model_name == "Stacking":
        model = context.second_level_models.get("Stacking")
        pred_cont = float(np.asarray(model.predict(meta_features)).ravel()[0])
    else:
        raise ValueError(f"Непідтримувана модель для регресії: {model_name}")

    pred_cls = int(0 if pred_cont <= context.q1 else 1 if pred_cont <= context.q3 else 2)
    return {"predicted_score": pred_cont, "predicted_class": pred_cls}


def interpret_predicted_class(pred_cls: int) -> str:
    mapping = {
        0: "Низький рейтинг / рівень успішності",
        1: "Середній рейтинг / рівень успішності",
        2: "Високий рейтинг / рівень успішності",
    }
    return mapping.get(int(pred_cls), f"Клас {pred_cls}")


def run_live_student_prediction(
    context: LivePredictionContext,
    raw_df: pd.DataFrame,
    model_name: str,
) -> dict[str, Any]:
    if model_name not in available_prediction_models(context):
        raise ValueError("Обрана модель відсутня серед доступних навчених моделей.")

    x_prepared = transform_student_row(context, raw_df)
    meta_features = build_live_meta_features(context, x_prepared)

    if context.strategy_type == CLASSIFICATION_STRATEGY:
        prediction = classification_from_model(context, model_name, x_prepared, meta_features)
    else:
        prediction = regression_from_model(context, model_name, x_prepared, meta_features)

    prediction.update(
        {
            "strategy_type": context.strategy_type,
            "strategy_label": strategy_label(context.strategy_type),
            "model_name": model_name,
            "model_label": model_display_name(model_name),
            "q1": context.q1,
            "q3": context.q3,
        }
    )
    return prediction


def render_prediction_result(prediction: dict[str, Any]) -> None:
    st.markdown("---")
    st.markdown("## Результат прогнозування")

    class_text = interpret_predicted_class(int(prediction["predicted_class"]))
    if prediction["strategy_type"] == CLASSIFICATION_STRATEGY:
        st.markdown(f"**Стратегія:** {prediction['strategy_label']}")
        st.markdown(f"**Використана модель:** {prediction['model_label']}")

        st.metric("Прогнозований клас", str(prediction["predicted_class"]))
        st.success(f"Інтерпретація прогнозу: **{class_text}**")
        if prediction.get("probabilities") is not None:
            probs = prediction["probabilities"]
            st.info(
                "Пояснення до класів прогнозу: "
                "0 — Low (низький рівень), "
                "1 — Medium (середній рівень), "
                "2 — High (високий рівень)."
            )
            prob_df = pd.DataFrame(
                {
                    "Клас": list(range(len(probs))),
                    "Ймовірність": [round(float(p), 4) for p in probs],
                }
            )
            st.dataframe(prob_df, use_container_width=True, hide_index=True)
        return

    st.markdown(f"**Поточна стратегія:** {prediction['strategy_label']}")
    st.markdown(f"**Використана модель:** {prediction['model_label']}")

    c1, c2 = st.columns(2)
    c1.metric("Прогнозований бал", metric_value(prediction["predicted_score"], digits=2))
    c2.metric("Прогнозований клас", str(prediction["predicted_class"]))
    st.success(f"Інтерпретація прогнозу: **{class_text}**")

    q1 = prediction.get("q1")
    q3 = prediction.get("q3")
    if q1 is not None and q3 is not None:
        st.caption(f"Категоризація виконується за порогами Q1 = {metric_value(q1, 2)} та Q3 = {metric_value(q3, 2)}")


def render_student_rating_prediction_tab() -> None:
    st.markdown("### Прогноз рейтингу студента")
    st.write("Ця вкладка використовує поточні навчені моделі та дозволяє виконати практичний прогноз для нового студента.")

    ready, missing = session_ready_for_prediction()
    if not ready:
        st.warning(
            "Практичний прогноз поки недоступний. Спочатку виконайте етапи попередньої обробки... "
        )
        return

    context = build_live_prediction_context()
    selected_model = render_prediction_model_selector(context)
    if selected_model is None:
        return

    if not context.feature_cols or context.x_raw is None or context.x_raw.empty:
        st.warning("Не знайдено вихідних ознак для побудови форми введення нового студента.")
        return

    raw_df = render_student_input_form(context.feature_cols, context.x_raw)
    if raw_df.empty:
        st.info("Заповніть форму вище та натисніть кнопку прогнозування.")
        return

    try:
        prediction = run_live_student_prediction(context, raw_df, selected_model)
    except Exception as exc:
        st.error(f"Не вдалося виконати прогнозування: {exc}")
        return

    st.session_state.module9_last_student_prediction = prediction
    render_prediction_result(prediction)
