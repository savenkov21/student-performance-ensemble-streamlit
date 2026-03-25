import streamlit as st, pandas as pd

from module_3 import _draw_panels, _numeric_columns

def _load_df(_file) -> pd.DataFrame:
    """Load a DataFrame from a file, trying different encodings if necessary."""
    try:
        df = pd.read_csv(_file, low_memory=False)
        return df
    except UnicodeDecodeError: 
        return pd.read_csv(_file, encoding='latin-1', low_memory=False)

def _reset_ml_state():
    """Очистити все, що пов'язано з ML-пайплайном, коли завантажили новий файл."""
    keys_to_remove = [
        "target_col", "task_type", "feature_cols",
        "X", "y",
        "X_train", "X_test", "y_train", "y_test",
        "test_size", "random_state", "stratify_used",
        "split_ready"
    ]
    for key in keys_to_remove:
        st.session_state.pop(key, None)

def page_analysis():
    st.subheader("Завантаження та аналіз даних")
    st.write("Завантажте CSV-файл та отримайте базовий огляд даних.")
    
    file = st.file_uploader("Виберіть CSV-файл", type=["csv"])
    
    if file is not None and ("df_orig" not in st.session_state or file.name != st.session_state.get("file_name")):
        df_loaded = _load_df(file)
        st.session_state.file_name = file.name
        st.session_state.df_orig = df_loaded.copy()
        st.session_state.df = df_loaded.copy()
        st.session_state.df_before = df_loaded.copy()
        st.session_state.log = []
        _reset_ml_state()
        st.success("Набір даних успішно завантажено!")
        
    if "df" in st.session_state:
        df = st.session_state.df
        
        st.markdown("### Попередній перегляд даних")
        st.dataframe(df.head(20), use_container_width=True)
        
        st.markdown("### Основна інформація")
        st.write(f"Кількість рядків: **{df.shape[0]}**, кількість стовпців: **{df.shape[1]}**")
        st.write(df.dtypes.to_frame("dtype"))
        
        st.markdown("### Швидка перевірка якості даних")
        missing = df.isna().sum()
        dup_count = df.duplicated().sum()
        rows_with_nan = df.isna().any(axis=1).sum()
        
        st.write(f"Рядків з пропущеними значеннями: {rows_with_nan}")
        st.write("Розподіл пропущених значень по стовпцях:")
        if (missing > 0).any():
            st.write(missing[missing > 0])
        else:
            st.info("Немає пропущених значень.")
        
        st.write(f"Кількість дублікатів: **{dup_count}**")
        
        st.markdown("### Статистичні показники числових змінних")
        num_cols = _numeric_columns(df)
        if num_cols:
            st.write(df[num_cols].describe().T)
        else:
            st.info("Числових колонок не знайдено.")
            
        if num_cols:
            st.markdown("### Візуалізація числових змінних")
            plot_type = st.selectbox("Тип графіка", options=["Histogram", "Boxplot", "Violin"], index=0)
            selected_cols = st.multiselect("Виберіть стовпці для графіків", options=num_cols, default=num_cols)
            if st.button("Побудувати графіки", use_container_width=True):
                _draw_panels(df, selected_cols, plot_type, mode_tag="before_norm")
            
    else:
        st.info("Поки що не завантажено жодного файлу. Використайте кнопку вище для завантаження.")
