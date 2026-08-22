import pandas as pd
import streamlit as st

from utils.constants_dataclasses import SPLIT_STATE_KEYS, SplitArtifacts


def invalidate_split_state() -> None:
    for key in SPLIT_STATE_KEYS:
        st.session_state.pop(key, None)


def save_split_to_session(artifacts: SplitArtifacts) -> None:
    data_to_store = {
        "strategy_type": artifacts.strategy_type,
        "target_col": artifacts.target_col,
        "feature_cols": artifacts.feature_cols,
        "feature_names_after_preprocessing": artifacts.feature_names_after_preprocessing,
        "preprocessing_config": artifacts.preprocessing_config,
        "preprocessor": artifacts.preprocessor,
        "X_raw": artifacts.X_raw.copy(),
        "X_train_raw": artifacts.X_train_raw.copy(),
        "X_test_raw": artifacts.X_test_raw.copy(),
        "X_train": artifacts.X_train.copy(),
        "X_test": artifacts.X_test.copy(),
        "X": pd.concat([artifacts.X_train, artifacts.X_test]).sort_index().copy(),
        "y": artifacts.y_active.copy(),
        "y_train": artifacts.y_train.copy(),
        "y_test": artifacts.y_test.copy(),
        "y_cont": artifacts.y_cont.copy(),
        "y_train_cont": artifacts.y_train_cont.copy(),
        "y_test_cont": artifacts.y_test_cont.copy(),
        "y_train_cls": artifacts.y_train_cls.copy(),
        "y_test_cls": artifacts.y_test_cls.copy(),
        "Q1": float(artifacts.q1),
        "Q3": float(artifacts.q3),
        "test_size": float(artifacts.test_size),
        "random_state": int(artifacts.random_state),
        "stratify_used": bool(artifacts.stratify_used),
        "split_ready": True,
    }
    st.session_state.update(data_to_store)
