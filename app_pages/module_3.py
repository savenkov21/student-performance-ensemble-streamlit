from __future__ import annotations

from io import BytesIO
import datetime

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from utils.visualization_helpers import draw_panels, numeric_columns


def page_visualization() -> None:
    if "df" not in st.session_state:
        st.warning("Спочатку завантажте дані на сторінці 'Завантаження та первинний аналіз даних'.")
        return

    df = st.session_state.df

    st.markdown("### Візуалізація розподілів")

    mode = st.radio(
        "Режим візуалізації:",
        options=["До нормалізації", "Після нормалізації"],
        horizontal=True,
    )

    plot_type = st.selectbox("Тип графіка", options=["Histogram", "Boxplot", "Violin"], index=0)

    num_cols = numeric_columns(df)
    selected_cols = st.multiselect(
        "Виберіть стовпці для візуалізації",
        options=num_cols,
        default=num_cols[:6],
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

        draw_panels(plot_df, selected_cols, plot_type, mode_tag)

    st.markdown("---")
    st.markdown("### Матриця кореляції")
    st.caption(
        "Матриця кореляції використовується для аналізу взаємозв’язків між ознаками. "
        "Для числових змінних застосовується коефіцієнт кореляції Пірсона, "
        "для порядкових змінних або монотонних нелінійних залежностей — коефіцієнт Спірмена, "
        "який є менш чутливим до викидів."
    )

    num_cols_all = numeric_columns(df)
    default_corr_cols = num_cols_all[:10] if len(num_cols_all) > 10 else num_cols_all

    corr_cols = st.multiselect(
        "Виберіть числові стовпці для кореляції:",
        options=num_cols_all,
        default=default_corr_cols,
    )

    max_rows = st.slider(
        "Макс. кількість рядків для кореляції (для продуктивності)",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100,
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
                annot_kws={"size": 15},
            )

            ax.set_title(f"{method} Correlation Matrix", fontsize=18)
            ax.tick_params(axis="x", labelsize=20)
            ax.tick_params(axis="y", labelsize=20)

            cbar = hm.collections[0].colorbar
            cbar.ax.tick_params(labelsize=18)
            st.pyplot(fig)

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
                use_container_width=True,
            )
