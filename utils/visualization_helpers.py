from __future__ import annotations

from math import ceil
from io import BytesIO
import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include="number").columns.tolist()


def draw_panels(df: pd.DataFrame, cols: list[str], kind: str, mode_tag: str) -> None:
    if not cols:
        st.info("Виберіть хоча б одну колонку.")
        return

    if mode_tag == "before_norm":
        color_hist = color_violin = color_box = "salmon"
    else:
        color_hist = color_violin = color_box = "skyblue"

    n = len(cols)
    ncols = 3
    nrows = ceil(n / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 7 * nrows))
    axes = axes.flatten() if n > 1 else [axes]

    title_fontsize = 28
    label_fontsize = 28
    tick_fontsize = 20

    for i, col in enumerate(cols):
        ax = axes[i]
        if kind == "Histogram":
            ax.hist(df[col].dropna(), bins=15, color=color_hist, edgecolor="black")
            ax.set_xlabel(col, fontsize=label_fontsize)
            ax.set_ylabel("Count", fontsize=label_fontsize)
            ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
        elif kind == "Violin":
            sns.violinplot(y=df[col], ax=ax, color=color_violin)
            ax.set_ylabel(col, fontsize=label_fontsize)
            ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
        elif kind == "Boxplot":
            sns.boxplot(y=df[col], ax=ax, color=color_box)
            ax.set_ylabel(col, fontsize=label_fontsize)
            ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)

    for j in range(i + 1, nrows * ncols):
        fig.delaxes(axes[j])

    fig.tight_layout(pad=7.0)
    st.pyplot(fig)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"plots_{kind.lower()}_{mode_tag}_{ts}.png"

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    buf.seek(0)

    st.download_button(
        label=f"Зберегти графік ({mode_tag})",
        data=buf,
        file_name=filename,
        mime="image/png",
        use_container_width=True,
    )
