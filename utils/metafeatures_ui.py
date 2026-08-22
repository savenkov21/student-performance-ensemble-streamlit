from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import streamlit as st

from utils.metafeatures_constants import (
    CLASSIFICATION_STRATEGY,
    MODEL_NAME_LABELS,
    REGRESSION_STRATEGY,
    STRATEGY_LABELS,
    MetaFeatureConfig,
    MetaFeatureContext,
)


def strategy_label(strategy_type: str) -> str:
    return STRATEGY_LABELS.get(strategy_type, strategy_type)


def short_model_name(name: str) -> str:
    return MODEL_NAME_LABELS.get(name, name)


def bool_label(value: bool) -> str:
    return "Так" if bool(value) else "Ні"


def metric(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def render_hero(context: MetaFeatureContext) -> None:
    ready = st.session_state.get("meta_ready") and st.session_state.get("meta_strategy_type") == context.strategy_type
    status_text = (
        "Метаознаки вже сформовані для поточної стратегії. Їх можна переглянути або перебудувати."
        if ready
        else "На цьому етапі будуються OOF-прогнози базових моделей першого рівня, які стануть вхідними ознаками для ансамблів другого рівня."
    )
    st.subheader("Формування метаознак")
    st.caption(status_text)


def render_overview_cards(context: MetaFeatureContext) -> None:
    st.subheader("1. Поточний стан даних")
    st.caption("Огляд конфігурації, яка буде використана для побудови OOF-метаознак.")

    st.markdown(f"**Стратегія:** {strategy_label(context.strategy_type)}")
    st.markdown(f"**Цільова змінна:** {context.target_col or '—'}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Навчальна вибірка", getattr(context.X_train, "shape", (0, 0))[0])
    c2.metric("Тестова вибірка", getattr(context.X_test, "shape", (0, 0))[0])
    c3.metric("Ознаки", getattr(context.X_train, "shape", (0, 0))[1])

    with st.expander("Показати деталі конфігурації", expanded=False):
        d1, d2, d3 = st.columns(3)
        d1.metric("Q1", metric(context.q1))
        d2.metric("Q3", metric(context.q3))
        d3.metric("Ознак 1-го рівня", str(pd.DataFrame(context.X_train).shape[1]))

        if context.strategy_type == CLASSIFICATION_STRATEGY:
            st.info("Для класифікаційної стратегії метаознаками стають імовірності класів із OOF-прогнозів базових моделей.")
        else:
            st.info("Для регресійної стратегії метаознаками стають неперервні OOF-прогнози базових моделей; за потреби додаються початкові ознаки.")

        st.markdown(
            "**Базові моделі:** Ridge, Random Forest, XGBoost  \n"
            "**Призначення етапу:** побудова стабільних метаознак без leakage."
        )


def render_theory_block() -> None:
    with st.expander("Коротке теоретичне пояснення", expanded=False):
        st.markdown(
            '''
            **OOF (out-of-fold) метаознаки** — це прогнози базових моделей, сформовані так,
            щоб метамодель не бачила прогнозів, отриманих на тих самих даних, на яких модель навчалась.

            **Логіка побудови:**
            1. Навчальна вибірка ділиться на `k` folds.
            2. На кожному кроці базова модель навчається на `k-1` folds.
            3. Потім вона прогнозує на відкладеному fold.
            4. Сукупність таких прогнозів формує метаознаки для `train`.
            5. Для `test` прогноз усереднюється між усіма fold-моделями.

            У класифікації метаознаками є **ймовірності класів**, а в регресії —
            **неперервні прогнози** базових моделей.
            '''
        )


def render_controls(default_cv: int = 5, default_passthrough: bool = False) -> tuple[MetaFeatureConfig, bool, bool]:
    st.subheader("2. Налаштування генерації")
    st.caption("Задайте параметри OOF-генерації та, за потреби, додайте вихідні ознаки першого рівня до метапростору.")

    left, right = st.columns([1.15, 0.85])

    with left:
        st.markdown("**Параметри генерації**")
        cv_folds = st.slider(
            "Кількість folds для OOF-прогнозів",
            min_value=3,
            max_value=10,
            value=int(st.session_state.get("meta_cv_folds", default_cv)),
            step=1,
            help="Більша кількість folds дає стабільніші OOF-ознаки, але збільшує час обчислень.",
        )
        passthrough = st.checkbox(
            "Додати початкові ознаки до метаознак (passthrough)",
            value=bool(st.session_state.get("meta_passthrough", default_passthrough)),
            help="До OOF-метаознак буде додано вихідні ознаки першого рівня.",
        )
        st.session_state.meta_cv_folds = int(cv_folds)
        st.session_state.meta_passthrough = bool(passthrough)

        st.caption("Метод побудови: OOF-прогнози базових моделей із k-fold розбиттям.")
        b1, b2 = st.columns([1.5, 1])
        build_clicked = b1.button("Сформувати метаознаки", use_container_width=True, type="primary")
        clear_clicked = b2.button("Очистити", use_container_width=True)

    with right:
        st.markdown("**Коротко про результат**")
        st.write("Після запуску буде сформовано `meta_X_train` для навчання метамоделі та `meta_X_test` для подальшого оцінювання ансамблів другого рівня.")
        st.write("У класифікації метаознаками є ймовірності класів, а в регресії — неперервні прогнози базових моделей.")
        with st.expander("Поточна конфігурація", expanded=False):
            c1, c2 = st.columns(2)
            c1.metric("CV folds", str(cv_folds))
            c2.metric("Passthrough", bool_label(bool(passthrough)))

    return MetaFeatureConfig(cv_folds=int(cv_folds), passthrough=bool(passthrough)), build_clicked, clear_clicked


def build_summary_payload() -> Dict[str, Any]:
    meta_x_train = st.session_state.get("meta_X_train")
    meta_x_test = st.session_state.get("meta_X_test")
    feature_names = st.session_state.get("meta_feature_names", [])
    params = st.session_state.get("meta_generation_params", {})
    base_models = st.session_state.get("meta_base_models", [])

    return {
        "meta_train_shape": getattr(meta_x_train, "shape", None),
        "meta_test_shape": getattr(meta_x_test, "shape", None),
        "feature_count": len(feature_names),
        "base_model_count": len(base_models),
        "cv_folds": params.get("cv_folds"),
        "passthrough": params.get("passthrough", False),
    }


def render_result_summary() -> None:
    meta_x_train = st.session_state.get("meta_X_train")
    meta_x_test = st.session_state.get("meta_X_test")
    if meta_x_train is None or meta_x_test is None:
        return

    summary = build_summary_payload()
    train_shape = summary["meta_train_shape"] or (0, 0)
    test_shape = summary["meta_test_shape"] or (0, 0)
    feature_names = st.session_state.get("meta_feature_names", [])
    base_models = st.session_state.get("meta_base_models", [])
    params = st.session_state.get("meta_generation_params", {})

    st.subheader("3. Перегляд результатів")
    st.caption("Підсумок сформованого метапростору та попередній перегляд матриць метаознак.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Навчальна вибірка", f"{train_shape[0]} × {train_shape[1]}")
    c2.metric("Тестова вибірка", f"{test_shape[0]} × {test_shape[1]}")
    c3.metric("К-ть метаознак", str(summary["feature_count"]))
    c4.metric("CV folds", str(summary["cv_folds"]))

    tab1, tab2 = st.tabs(["Preview train/test", "Структура метаознак"])

    with tab1:
        left, right = st.columns(2)
        with left:
            st.markdown(f"**meta_X_train:** `{meta_x_train.shape}`")
            st.dataframe(meta_x_train.head(10), use_container_width=True)
        with right:
            st.markdown(f"**meta_X_test:** `{meta_x_test.shape}`")
            st.dataframe(meta_x_test.head(10), use_container_width=True)

    with tab2:
        preview_df = pd.DataFrame(
            {
                "Метаознака": feature_names,
                "Тип блоку": [
                    "Probability / прогноз" if str(name).startswith("meta_") else "Passthrough"
                    for name in feature_names
                ],
            }
        )
        st.dataframe(
            preview_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Метаознака": st.column_config.TextColumn("Метаознака", width="large"),
                "Тип блоку": st.column_config.TextColumn("Тип блоку", width="medium"),
            },
        )
