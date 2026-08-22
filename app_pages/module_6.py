from __future__ import annotations

import streamlit as st

from utils.second_level_core import run_training
from utils.second_level_state import check_ready_for_second_level, clear_second_level_state, load_context, save_artifacts
from utils.second_level_ui import render_controls, render_experiment_snapshot, render_hero, render_results


def page_training_second_level_model():
    if not check_ready_for_second_level():
        return

    context = load_context()
    render_hero(context)
    render_experiment_snapshot(context)
    config, train_clicked, clear_clicked = render_controls(context)

    if clear_clicked:
        clear_second_level_state()
        st.success("Результати другого рівня очищено.")

    if train_clicked:
        try:
            with st.spinner("Триває навчання ансамблів другого рівня..."):
                artifacts = run_training(context, config)
                save_artifacts(context, artifacts)
            st.success("Ансамблі другого рівня успішно навчені.")
        except ValueError as exc:
            st.error(f"Помилка під час навчання: {exc}")

    render_results(context)
