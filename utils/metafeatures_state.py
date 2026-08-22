from __future__ import annotations

import streamlit as st

from utils.metafeatures_constants import (
    META_STATE_KEYS,
    REQUIRED_STATE_KEYS,
    MetaFeatureArtifacts,
    MetaFeatureContext,
)


def check_ready_for_metafeatures() -> bool:
    missing = [key for key in REQUIRED_STATE_KEYS if key not in st.session_state]
    if missing:
        st.warning("Спочатку завантажте дані на сторінці 'Завантаження та первинний аналіз даних'.")
        return False

    if st.session_state.get("first_level_strategy_type") != st.session_state.get("strategy_type"):
        st.warning("Стратегія змінилася. Перенавчіть базові моделі на сторінці 'Навчання базових моделей'.")
        return False

    return True


def clear_metafeatures_state() -> None:
    for key in META_STATE_KEYS:
        st.session_state.pop(key, None)


def load_context() -> MetaFeatureContext:
    return MetaFeatureContext(
        X_train=st.session_state.X_train,
        X_test=st.session_state.X_test,
        y_train=st.session_state.get("y_train"),
        y_train_cont=st.session_state.get("y_train_cont"),
        strategy_type=st.session_state.strategy_type,
        params=st.session_state.first_level_params,
        q1=st.session_state.get("Q1"),
        q3=st.session_state.get("Q3"),
        target_col=st.session_state.get("target_col"),
    )


def save_metafeatures(artifacts: MetaFeatureArtifacts, strategy_type: str) -> None:
    state_payload = {
        "meta_X_train": artifacts.meta_X_train,
        "meta_X_test": artifacts.meta_X_test,
        "meta_feature_names": artifacts.feature_names,
        "meta_generation_params": artifacts.generation_params,
        "meta_base_models": artifacts.base_model_names,
        "meta_full_train_models": artifacts.full_train_models,
        "meta_ready": True,
        "meta_strategy_type": strategy_type,
    }
    for key, value in state_payload.items():
        st.session_state[key] = value
