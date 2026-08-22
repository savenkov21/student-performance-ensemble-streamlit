import streamlit as st

from utils.preprocessing_ui import (
    render_cleaning_panel,
    render_dataset_overview,
    render_dataset_preview,
    render_header,
    render_split_actions,
    render_split_summary,
    render_strategy_form,
)


def page_preprocessing() -> None:
    if "df" not in st.session_state:
        st.warning("Спочатку завантажте дані на сторінці 'Завантаження та первинний аналіз даних'.")
        return

    render_header()

    if "df_before" not in st.session_state:
        st.session_state.df_before = st.session_state.df.copy()

    df_current = st.session_state.df.copy()

    render_dataset_overview(df_current)
    render_cleaning_panel()
    render_dataset_preview(st.session_state.df.copy())
    config = render_strategy_form(st.session_state.df.copy())
    render_split_actions(st.session_state.df.copy(), config)
    render_split_summary()
