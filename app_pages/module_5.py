from __future__ import annotations

import streamlit as st

from utils.metafeatures_core import build_metafeatures
from utils.metafeatures_state import (
    check_ready_for_metafeatures,
    clear_metafeatures_state,
    load_context,
    save_metafeatures,
)
from utils.metafeatures_ui import (
    render_controls,
    render_hero,
    render_overview_cards,
    render_result_summary,
    render_theory_block,
)


def page_formation_metafeatures() -> None:
    if not check_ready_for_metafeatures():
        return

    context = load_context()
    render_hero(context)
    render_overview_cards(context)
    render_theory_block()

    config, build_clicked, clear_clicked = render_controls()

    if clear_clicked:
        clear_metafeatures_state()
        st.success("Метаознаки очищено.")
        return

    if build_clicked:
        with st.spinner("Триває формування OOF-метаознак..."):
            artifacts = build_metafeatures(context=context, config=config)
            save_metafeatures(artifacts, strategy_type=context.strategy_type)
        st.success("Метаознаки успішно сформовано!")

    if st.session_state.get("meta_strategy_type") != context.strategy_type:
        st.warning("Стратегія змінилася. Сформуйте метаознаки повторно.")
        return

    if not st.session_state.get("meta_ready", False):
        return

    render_result_summary()
