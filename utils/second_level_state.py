from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.second_level_constants import (
    CLASSIFICATION_STRATEGY,
    REQUIRED_META_KEYS,
    SECOND_LEVEL_STATE_KEYS,
    SecondLevelArtifacts,
    SecondLevelContext,
)


def check_ready_for_second_level() -> bool:
    missing = [k for k in REQUIRED_META_KEYS if k not in st.session_state]
    if missing:
        st.warning("Спочатку завантажте дані на сторінці 'Завантаження та первинний аналіз даних'.")
        return False

    if not st.session_state.get("meta_ready", False):
        st.warning("Спочатку виконайте формування метаознак на сторінці 'Формування метаознак'.")
        return False

    if st.session_state.get("meta_strategy_type") != st.session_state.get("strategy_type"):
        st.warning("Стратегія змінилася. Сформуйте метаознаки повторно.")
        return False

    strategy_type = st.session_state.get("strategy_type")
    if strategy_type == CLASSIFICATION_STRATEGY:
        needed = ["y_train", "y_test"]
    else:
        needed = ["y_train_cont", "y_test_cont"]
    strategy_missing = [k for k in needed if k not in st.session_state]
    if strategy_missing:
        st.warning("Немає потрібних цільових змінних для навчання другого рівня. Виконайте попередні етапи повторно.")
        return False

    return True


def clear_second_level_state() -> None:
    for key in SECOND_LEVEL_STATE_KEYS:
        st.session_state.pop(key, None)


def load_context() -> SecondLevelContext:
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


def save_artifacts(context: SecondLevelContext, artifacts: SecondLevelArtifacts) -> None:
    payload = {
        "second_level_models": artifacts.models,
        "second_level_results": artifacts.results,
        "second_level_predictions": artifacts.predictions,
        "second_level_predictions_cls": artifacts.predictions_cls,
        "second_level_params": artifacts.params,
        "second_level_strategy_type": context.strategy_type,
    }
    for key, value in payload.items():
        st.session_state[key] = value
