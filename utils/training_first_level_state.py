from __future__ import annotations

import streamlit as st

from utils.training_first_level_constants import (
    FIRST_LEVEL_STATE_KEYS,
    REQUIRED_SPLIT_KEYS,
    TrainingArtifacts,
    TrainingContext,
)


def clear_first_level_state() -> None:
    for key in FIRST_LEVEL_STATE_KEYS:
        st.session_state.pop(key, None)


def check_split_ready() -> bool:
    missing = [key for key in REQUIRED_SPLIT_KEYS if key not in st.session_state]
    if missing:
        st.warning("Спочатку завантажте дані на сторінці 'Завантаження та первинний аналіз даних'.")
        return False
    if not st.session_state.get("split_ready", False):
        st.warning("Спочатку виконайте train/test split на сторінці 'Попередня обробка'.")
        return False
    return True


def build_training_context() -> TrainingContext:
    return TrainingContext(
        X_train=st.session_state.X_train,
        X_test=st.session_state.X_test,
        strategy_type=st.session_state.strategy_type,
        q1=float(st.session_state.Q1),
        q3=float(st.session_state.Q3),
        y_train=st.session_state.y_train,
        y_test=st.session_state.y_test,
        y_train_cont=st.session_state.y_train_cont,
        y_test_cont=st.session_state.y_test_cont,
        target_col=st.session_state.get("target_col", "—"),
    )


def save_training_artifacts(context: TrainingContext, artifacts: TrainingArtifacts) -> None:
    st.session_state.first_level_models = artifacts.models
    st.session_state.first_level_results = artifacts.results
    st.session_state.first_level_predictions = artifacts.predictions
    st.session_state.first_level_predictions_cls = artifacts.predictions_cls
    st.session_state.first_level_params = artifacts.params
    st.session_state.first_level_strategy_type = context.strategy_type
    st.session_state.optimization_mode = "bayesian_optimization"
