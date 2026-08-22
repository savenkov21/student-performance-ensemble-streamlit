from __future__ import annotations

import streamlit as st

from utils.training_first_level_core import run_first_level_training, run_optimization
from utils.training_first_level_state import build_training_context, check_split_ready, clear_first_level_state
from utils.training_first_level_ui import (
    render_bayesian_block,
    render_data_snapshot,
    render_optimization_results,
    render_page_intro,
    render_results_if_ready,
    render_training_controls,
)


def page_training_first_level_models() -> None:
    if not check_split_ready():
        return

    context = build_training_context()
    render_page_intro(context)
    render_data_snapshot(context)

    optimization_config, optimize_clicked, clear_opt_clicked = render_bayesian_block(context.strategy_type)

    # Always reserve exactly one place for optimization status messages.
    # This keeps the Streamlit element tree stable between reruns and prevents
    # stale elements below this point from being shown as visual duplicates.
    optimization_status = st.empty()

    if clear_opt_clicked:
        st.session_state.pop("optimization_results", None)
        with optimization_status.container():
            st.success("Результати оптимізації очищено.")

    if optimize_clicked:
        with optimization_status.container():
            run_optimization(context, optimization_config)

    optimization_results = st.session_state.get("optimization_results")
    render_optimization_results(optimization_results, context.strategy_type)

    train_clicked, clear_train_clicked = render_training_controls()

    # The training spinner/success message is also isolated in a permanent
    # placeholder. Dynamic status UI therefore cannot shift the results block.
    training_status = st.empty()

    if clear_train_clicked:
        clear_first_level_state()
        with training_status.container():
            st.success("Результати базових моделей очищено.")

    if train_clicked:
        with training_status.container():
            run_first_level_training(context, optimization_results)

    render_results_if_ready(context)
