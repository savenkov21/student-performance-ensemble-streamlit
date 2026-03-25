#module_3.py

import streamlit as st, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, numpy as np

from math import ceil
from io import BytesIO
import datetime

#---------------- Auxiliary functions for visualization ------------------
def _numeric_columns(df: pd.DataFrame):
    """Return list of numeric columns in the DataFrame."""
    return df.select_dtypes(include="number").columns.tolist()

def _draw_panels(df: pd.DataFrame, cols: list[str], kind: str, mode_tag: str):
    if not cols:
        st.info("Виберіть хоча б одну колонку.")
        return

    if mode_tag == "before_norm":
        color_hist = "salmon"
        color_violin = "salmon"
        color_box = "salmon"
    else:  # after_norm
        color_hist = "skyblue"
        color_violin = "skyblue"
        color_box = "skyblue"

    n = len(cols)
    ncols = 3
    nrows = ceil(n / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 7*nrows)) 
    axes = axes.flatten() if n > 1 else [axes]

    title_fontsize = 28   # graph titles
    label_fontsize = 28   # axis labels
    tick_fontsize = 20    # tick labels

    for i, col in enumerate(cols): 
        ax = axes[i] 
        if kind == "Histogram": 
            ax.hist(df[col].dropna(), bins=15, color=color_hist, edgecolor="black") 
            #ax.set_title(f"Histogram: {col}", fontsize=title_fontsize) 
            ax.set_xlabel(col, fontsize=label_fontsize) 
            ax.set_ylabel("Count", fontsize=label_fontsize) 
            ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        elif kind == "Violin":
            sns.violinplot(y=df[col], ax=ax, color=color_violin) 
            #ax.set_title(f"Violin: {col}", fontsize=title_fontsize) 
            ax.set_ylabel(col, fontsize=label_fontsize) 
            ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        elif kind == "Boxplot": 
            sns.boxplot(y=df[col], ax=ax, color=color_box) 
            #ax.set_title(f"Boxplot: {col}", fontsize=title_fontsize) 
            ax.set_ylabel(col, fontsize=label_fontsize)
            ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

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
        use_container_width=True
    )

#------------------------ Main page function ------------------------
def page_visualization():

    if "df" not in st.session_state:
        st.warning("Спочатку завантажте дані на сторінці 'Завантаження та аналіз даних'.")
        return

    df = st.session_state.df

    st.markdown("### Візуалізація розподілів")

    mode = st.radio("Режим візуалізації:",
        options=["До нормалізації", "Після нормалізації"],
        horizontal=True
    )

    plot_type = st.selectbox("Тип графіка",
        options=["Histogram", "Boxplot", "Violin"],
        index=0
    )

    num_cols = _numeric_columns(df)
    selected_cols = st.multiselect("Виберіть стовпці для візуалізації",
        options=num_cols,
        default=num_cols[:6]
    )

    if len(selected_cols) > 10:
        st.warning("Виберіть не більше 10 стовпців для кращої візуалізації.")

    if st.button("Побудувати графіки", use_container_width=True):
        if mode == "До нормалізації":
            if "df_before" not in st.session_state:
                st.warning("Дані до нормалізації ще не збережені.")
                return
            plot_df = st.session_state.df_before
            mode_tag = "before_norm"
        else:
            plot_df = df
            mode_tag = "after_norm"

        _draw_panels(plot_df, selected_cols, plot_type, mode_tag)



    st.markdown("---")
    st.markdown("### Матриця кореляції")
    st.caption(
        "Матриця кореляції використовується для аналізу взаємозв’язків між ознаками. "
        "Для числових змінних застосовується коефіцієнт кореляції Пірсона, "
        "для порядкових змінних або монотонних нелінійних залежностей — коефіцієнт Спірмена, "
        "який є менш чутливим до викидів."
    )

    num_cols_all = _numeric_columns(df)
    default_corr_cols = num_cols_all[:10] if len(num_cols_all) > 10 else num_cols_all

    corr_cols = st.multiselect("Виберіть числові стовпці для кореляції:", 
        options=num_cols_all, 
        default=default_corr_cols
    )

    max_rows = st.slider("Макс. кількість рядків для кореляції (для продуктивності)", 
        min_value=100, 
        max_value=10000, 
        value=1000, 
        step=100
    )

    method = st.selectbox("Метод кореляції", options=["Pearson", "Spearman"], index=0)

    if st.button("Показати матрицю кореляції", use_container_width=True):
        if len(corr_cols) < 2:
            st.warning("Виберіть принаймні 2 стовпці для кореляції.")
        else:
            plot_df = df.copy()
            if len(plot_df) > max_rows:
                plot_df = plot_df.sample(max_rows, random_state=42)

            corr_method = "pearson" if method == "Pearson" else "spearman"
            corr_matrix = plot_df[corr_cols].corr(method=corr_method)

            fig, ax = plt.subplots(figsize=(10, 8))
            hm = sns.heatmap(
                corr_matrix,
                annot=True,
                cmap="coolwarm",
                vmin=-1,
                vmax=1,
                ax=ax,
                annot_kws={"size": 15}  # розмір тексту всередині комірок
            )

            ax.set_title(f"{method} Correlation Matrix", fontsize=18)  # розмір заголовку
            ax.tick_params(axis='x', labelsize=20)  # розмір підписів по X
            ax.tick_params(axis='y', labelsize=20)  # розмір підписів по Y
            
            cbar = hm.collections[0].colorbar
            cbar.ax.tick_params(labelsize=18)   # розмір чисел шкали
            st.pyplot(fig)


            # Save the correlation matrix plot as a PNG file
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"correlation_{corr_method}_{ts}.png"

            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=150)
            buf.seek(0)

            st.download_button(
                label=f"Зберегти кореляційну матрицю ({method})",
                data=buf,
                file_name=filename,
                mime="image/png",
                use_container_width=True
            )