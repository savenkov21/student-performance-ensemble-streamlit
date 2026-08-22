from __future__ import annotations

from typing import Optional

import streamlit as st

from utils.evaluation_constants import EvaluationContext, REQUIRED_STATE_KEYS


def check_ready_for_evaluation() -> bool:
    missing = [key for key in REQUIRED_STATE_KEYS if key not in st.session_state]
    if missing:
        st.warning("Спочатку завантажте дані на сторінці 'Завантаження та первинний аналіз даних'.")
        return False

    strategy_type = st.session_state.get("strategy_type")
    if st.session_state.get("first_level_strategy_type") != strategy_type:
        st.warning("Стратегія змінилася. Перенавчіть базові моделі на попередньому етапі.")
        return False

    if st.session_state.get("second_level_strategy_type") != strategy_type:
        st.warning("Стратегія змінилася. Перенавчіть ансамблі другого рівня.")
        return False

    return True


def load_context() -> Optional[EvaluationContext]:
    if not check_ready_for_evaluation():
        return None

    return EvaluationContext(
        strategy_type=st.session_state.get("strategy_type"),
        target_col=st.session_state.get("target_col", "—"),
        q1=st.session_state.get("Q1", "—"),
        q3=st.session_state.get("Q3", "—"),
        meta_ready=bool(st.session_state.get("meta_ready", False)),
        first_level_results=st.session_state.get("first_level_results", {}),
        second_level_results=st.session_state.get("second_level_results", {}),
        first_level_params=st.session_state.get("first_level_params", {}),
        meta_generation_params=st.session_state.get("meta_generation_params", {}),
        second_level_params=st.session_state.get("second_level_params", {}),
        y_test_cls=st.session_state.get("y_test_cls"),
        test_sample_ids=(
            st.session_state.get("X_test_raw").index.tolist()
            if st.session_state.get("X_test_raw") is not None
            else st.session_state.get("X_test").index.tolist()
            if st.session_state.get("X_test") is not None
            else []
        ),
        test_features=st.session_state.get("X_test_raw", st.session_state.get("X_test")),
        first_level_predictions_cls=st.session_state.get("first_level_predictions_cls", {}),
        second_level_predictions_cls=st.session_state.get("second_level_predictions_cls", {}),
    )
