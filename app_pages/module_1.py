from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.app_state import reset_all_pipeline_state
from utils.visualization_helpers import draw_panels, numeric_columns


def load_df(file_obj) -> pd.DataFrame:
    try:
        return pd.read_csv(file_obj, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(file_obj, encoding="latin-1", low_memory=False)


def page_analysis() -> None:
    st.subheader("Завантаження та первинний аналіз даних")
    st.write("Завантажте CSV-файл та отримайте базовий огляд даних.")

    file = st.file_uploader("Виберіть CSV-файл", type=["csv"])

    if file is not None and ("df_orig" not in st.session_state or file.name != st.session_state.get("file_name")):
        df_loaded = load_df(file)
        st.session_state.file_name = file.name
        st.session_state.df_orig = df_loaded.copy()
        st.session_state.df = df_loaded.copy()
        st.session_state.df_before = df_loaded.copy()
        st.session_state.log = []
        reset_all_pipeline_state()
        st.success("Набір даних успішно завантажено!")

    if "df" not in st.session_state:
        st.info("Поки що не завантажено жодного файлу. Використайте кнопку вище для завантаження.")
        return

    df = st.session_state.df

    st.markdown("### Попередній перегляд даних")
    st.dataframe(df.head(20), use_container_width=True)

    st.markdown("### Основна інформація")
    st.write(f"Кількість рядків: **{df.shape[0]}**, кількість стовпців: **{df.shape[1]}**")
    st.write(df.dtypes.to_frame("dtype"))

    st.markdown("### Швидка перевірка якості даних")
    missing = df.isna().sum()
    dup_count = int(df.duplicated().sum())
    rows_with_nan = int(df.isna().any(axis=1).sum())

    st.write(f"Рядків з пропущеними значеннями: {rows_with_nan}")
    st.write("Розподіл пропущених значень по стовпцях:")
    if (missing > 0).any():
        st.write(missing[missing > 0])
    else:
        st.info("Немає пропущених значень.")

    st.write(f"Кількість дублікатів: **{dup_count}**")

    st.markdown("### Статистичні показники числових змінних")
    num_cols = numeric_columns(df)
    if num_cols:
        st.write(df[num_cols].describe().T)
    else:
        st.info("Числових колонок не знайдено.")

    if num_cols:
        st.markdown("### Візуалізація числових змінних")
        plot_type = st.selectbox("Тип графіка", options=["Histogram", "Boxplot", "Violin"], index=0)
        selected_cols = st.multiselect("Виберіть стовпці для графіків", options=num_cols, default=num_cols)
        if st.button("Побудувати графіки", use_container_width=True):
            draw_panels(df, selected_cols, plot_type, mode_tag="before_norm")
