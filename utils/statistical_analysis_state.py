from __future__ import annotations

from typing import Any, Dict, Optional

import streamlit as st


ARTIFACTS_KEY = "statistical_analysis_report"
REQUEST_KEY = "statistical_analysis_request_signature"


def save_statistical_analysis(report: Dict[str, Any], request_signature: str) -> None:
    st.session_state[ARTIFACTS_KEY] = report
    st.session_state[REQUEST_KEY] = request_signature


def load_statistical_analysis(request_signature: str) -> Optional[Dict[str, Any]]:
    if st.session_state.get(REQUEST_KEY) != request_signature:
        return None
    report = st.session_state.get(ARTIFACTS_KEY)
    return report if isinstance(report, dict) else None


def clear_statistical_analysis() -> None:
    st.session_state.pop(ARTIFACTS_KEY, None)
    st.session_state.pop(REQUEST_KEY, None)
