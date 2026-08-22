from __future__ import annotations

import ast
from typing import Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


def to_numeric_array(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        arr = value.astype(float, copy=False).ravel()
        return arr if arr.size else None
    if isinstance(value, (list, tuple)):
        try:
            arr = np.asarray(value, dtype=float).ravel()
            return arr if arr.size else None
        except Exception:
            return None
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except Exception:
            return None
        return to_numeric_array(parsed)
    return None


def plot_confusion_matrix(cm: np.ndarray, class_labels: List[Any], title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(class_labels)))
    ax.set_yticks(np.arange(len(class_labels)))
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)

    thresh = cm.max() / 2 if cm.size > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(int(cm[i, j]), "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolor="k")

    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)

    ax.set_xlabel("Actual values")
    ax.set_ylabel("Predicted values")
    ax.set_title(title)
    ax.grid(True)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_confusion_matrix_from_result(result: dict[str, Any], title: str, regression_mode: bool = False) -> None:
    if regression_mode:
        cm = result.get("confusion_matrix_after_discretization")
        labels = result.get("classes_after_discretization")
        if cm is None:
            cm = result.get("confusion_matrix")
        if labels is None:
            labels = result.get("classes")
    else:
        cm = result.get("confusion_matrix")
        labels = result.get("classes")

    if cm is None:
        return

    try:
        cm_np = np.asarray(cm, dtype=int)
    except Exception:
        st.caption("Не вдалося побудувати confusion matrix: некоректний формат матриці у JSON.")
        return

    if cm_np.ndim != 2 or cm_np.shape[0] != cm_np.shape[1]:
        st.caption("Не вдалося побудувати confusion matrix: очікується квадратна матриця.")
        return

    if not labels or len(labels) != cm_np.shape[0]:
        labels = list(range(cm_np.shape[0]))

    plot_confusion_matrix(cm_np, list(labels), title)


def render_actual_vs_predicted_from_result(result: dict[str, Any], title: str) -> None:
    true_keys = ["y_true_cont", "actual_values", "y_true", "actual"]
    pred_keys = ["y_pred_cont", "predicted_values", "y_pred", "predicted"]

    y_true = next((to_numeric_array(result.get(key)) for key in true_keys if key in result), None)
    y_pred = next((to_numeric_array(result.get(key)) for key in pred_keys if key in result), None)

    if y_true is None or y_pred is None:
        st.caption(
            "Графік Actual vs Predicted недоступний для цього JSON: у файлі немає збережених масивів "
            "фактичних і прогнозованих безперервних значень."
        )
        return

    if len(y_true) != len(y_pred) or len(y_true) == 0:
        st.caption("Не вдалося побудувати Actual vs Predicted: масиви мають різну довжину або порожні.")
        return

    plot_actual_vs_predicted(y_true, y_pred, title)
