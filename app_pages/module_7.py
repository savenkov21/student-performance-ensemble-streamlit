from __future__ import annotations

from utils.evaluation_core import build_artifacts
from utils.evaluation_state import load_context
from utils.evaluation_ui import (
    render_comparison_section,
    render_details_section,
    render_experiment_snapshot,
    render_export_section,
    render_page_header,
)


def page_evaluation_results() -> None:
    context = load_context()
    if context is None:
        return

    artifacts = build_artifacts(context)
    render_page_header(context)
    render_experiment_snapshot(context)
    render_comparison_section(context, artifacts)
    render_details_section(context)
    render_export_section(context, artifacts)
