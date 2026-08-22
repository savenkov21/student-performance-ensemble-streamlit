import pandas as pd
import streamlit as st

from utils.constants_dataclasses import (
    CATEGORICAL_IMPUTER_LABELS,
    CATEGORICAL_IMPUTER_LABELS_REVERSE,
    DEFAULT_PREPROCESSING_CONFIG,
    NUMERIC_IMPUTER_LABELS,
    NUMERIC_IMPUTER_LABELS_REVERSE,
    ONEHOT_ENCODING_LABELS,
    ONEHOT_ENCODING_LABELS_REVERSE,
    STRATEGY_LABELS,
    SplitConfig,
)
from utils.preprocessing_core import build_split_artifacts, categorical_columns, numeric_columns
from utils.session_state import invalidate_split_state, save_split_to_session


def render_header() -> None:
    st.subheader("Попередня обробка даних та формування вибірок")
    st.write(
        "На цьому етапі виконується вибір цільової змінної, очищення та трансформація даних, "
        "а також формування навчальної і тестової вибірок для подальшого моделювання."
    )
    st.caption("Очищення даних • Трансформація даних • Розбиття на train/test")


def render_dataset_overview(df: pd.DataFrame) -> None:
    rows, cols = df.shape
    rows_with_nan = int(df.isna().any(axis=1).sum())
    dup_count = int(df.duplicated().sum())
    numeric_count = len(numeric_columns(df))
    categorical_count = len(categorical_columns(df))

    st.markdown("### 1. Поточний набір даних")
    st.caption("Швидкий огляд структури датасету перед очищенням, трансформацією та розбиттям на вибірки.")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Рядки", rows)
    c2.metric("Стовпці", cols)
    c3.metric("Числові", numeric_count)
    c4.metric("Категоріальні", categorical_count)
    c5.metric("Рядки з NaN", rows_with_nan)
    c6.metric("Дублікати", dup_count)

    if rows_with_nan > 0 or dup_count > 0:
        st.warning(
            "У наборі даних є пропуски або дублікати. Їх бажано опрацювати перед формуванням навчальної та тестової вибірок."
        )
    else:
        st.success("Базова перевірка пройдена: явних пропусків у рядках і дублікатів не знайдено.")


def render_cleaning_panel() -> None:
    st.markdown("### 2. Очищення набору даних")
    st.caption("Видалення дублікатів, відновлення набору даних та параметри обробки пропущених значень.")

    with st.container(border=True):
        st.markdown("**Видалення дублікатів**")
        a1, a2 = st.columns(2)
        with a1:
            if st.button("Видалити дублікати", use_container_width=True):
                before = len(st.session_state.df)
                st.session_state.df = st.session_state.df.drop_duplicates().reset_index(drop=True)
                invalidate_split_state()
                st.success(f"Видалено дублікатів: {before - len(st.session_state.df)}")
        with a2:
            if st.button("Відновити", use_container_width=True):
                st.session_state.df = st.session_state.df_orig.copy()
                st.session_state.df_before = st.session_state.df_orig.copy()
                invalidate_split_state()
                st.success("Оригінальний набір даних відновлено.")

    with st.container(border=True):
        st.markdown("**Обробка пропущених значень**")

        df_current = st.session_state.df
        has_missing_rows = bool(df_current.isna().any(axis=1).sum())

        default_num_imputer = st.session_state.get("preprocessing_config", {}).get(
            "numeric_imputer_strategy", DEFAULT_PREPROCESSING_CONFIG["numeric_imputer_strategy"]
        )
        default_num_label = NUMERIC_IMPUTER_LABELS_REVERSE.get(default_num_imputer, "Середнє значення")

        default_cat_imputer = st.session_state.get("preprocessing_config", {}).get(
            "categorical_imputer_strategy", DEFAULT_PREPROCESSING_CONFIG["categorical_imputer_strategy"]
        )
        default_cat_label = CATEGORICAL_IMPUTER_LABELS_REVERSE.get(default_cat_imputer, "Найчастіше значення")

        if has_missing_rows:
            st.selectbox(
                "Заповнення пропусків у числових ознаках",
                options=list(NUMERIC_IMPUTER_LABELS.keys()),
                index=list(NUMERIC_IMPUTER_LABELS.keys()).index(default_num_label),
                key="numeric_imputer_label",
            )
            st.selectbox(
                "Заповнення пропусків у категоріальних ознаках",
                options=list(CATEGORICAL_IMPUTER_LABELS.keys()),
                index=list(CATEGORICAL_IMPUTER_LABELS.keys()).index(default_cat_label),
                key="categorical_imputer_label",
            )
            st.caption("Після зміни параметрів імпутації повторно сформуйте train/test split, щоб оновити preprocessing pipeline.")
        else:
            st.info(
                "У поточному наборі даних пропущених значень немає, тому окреме налаштування імпутації тут не потрібне."
            )
            st.session_state["numeric_imputer_label"] = default_num_label
            st.session_state["categorical_imputer_label"] = default_cat_label
            st.selectbox(
                "Заповнення пропусків у числових ознаках",
                options=["Не застосовується"],
                index=0,
                disabled=True,
                key="numeric_imputer_label_disabled",
            )
            st.selectbox(
                "Заповнення пропусків у категоріальних ознаках",
                options=["Не застосовується"],
                index=0,
                disabled=True,
                key="categorical_imputer_label_disabled",
            )


def render_dataset_preview(df: pd.DataFrame) -> None:
    st.markdown("### 3. Попередній перегляд даних")
    st.caption("Перевірка вмісту таблиці, типів стовпців і можливість швидко оцінити поточний датасет.")

    preview_tab1, preview_tab2 = st.tabs(["Перші рядки", "Типи стовпців"])

    with preview_tab1:
        st.dataframe(df.head(20), use_container_width=True)

    with preview_tab2:
        dtype_df = pd.DataFrame(
            {
                "column": df.columns,
                "dtype": df.dtypes.astype(str).values,
                "missing": df.isna().sum().values,
                "unique": df.nunique(dropna=False).values,
            }
        )
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)


def get_default_target(all_cols: list[str]) -> str:
    state_target = st.session_state.get("target_col")
    if state_target in all_cols:
        return state_target
    return all_cols[-1]


def get_default_strategy_index() -> int:
    current = st.session_state.get("strategy_type", "direct_multiclass_classification")
    return 0 if current == "direct_multiclass_classification" else 1


def render_strategy_form(df_current: pd.DataFrame) -> SplitConfig | None:
    st.markdown("### 4. Налаштування стратегії дослідження та трансформація даних")
    st.caption(
        "Вибір цільової ознаки, сценарію дослідження, параметрів train/test split та налаштування трансформації ознак."
    )

    all_cols = df_current.columns.tolist()
    if len(all_cols) < 2:
        st.warning("У наборі даних замало стовпців для формування X і y.")
        return None

    with st.container(border=True):
        st.markdown("**Конфігурація експерименту**")
        target_col = st.selectbox(
            "Цільова ознака (target)",
            options=all_cols,
            index=all_cols.index(get_default_target(all_cols)),
        )
        strategy_type = st.radio(
            "Стратегія дослідження",
            options=list(STRATEGY_LABELS.keys()),
            index=get_default_strategy_index(),
            format_func=lambda x: STRATEGY_LABELS[x],
        )
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            test_size = st.slider(
                "Частка тестової вибірки",
                min_value=0.10,
                max_value=0.40,
                value=float(st.session_state.get("test_size", DEFAULT_PREPROCESSING_CONFIG["test_size"])),
                step=0.05,
            )
        with col_t2:
            random_state = st.number_input(
                "Початкове значення random state",
                min_value=0,
                max_value=9999,
                value=int(st.session_state.get("random_state", DEFAULT_PREPROCESSING_CONFIG["random_state"])),
                step=1,
            )

    with st.container(border=True):
        st.markdown("**Трансформація даних**")
        scaler_type = st.radio(
            "Масштабування числових ознак",
            options=["None", "StandardScaler", "MinMaxScaler"],
            index=["None", "StandardScaler", "MinMaxScaler"].index(
                st.session_state.get("preprocessing_config", {}).get(
                    "scaler_type", DEFAULT_PREPROCESSING_CONFIG["scaler_type"]
                )
            ),
            horizontal=True,
        )
        st.info("Масштабування впливає лише на числові ознаки й виконується після обробки пропущених значень.")

        numeric_imputer_label = st.session_state.get("numeric_imputer_label", "Середнє значення")
        categorical_imputer_label = st.session_state.get("categorical_imputer_label", "Найчастіше значення")
        numeric_imputer_strategy = NUMERIC_IMPUTER_LABELS[numeric_imputer_label]
        categorical_imputer_strategy = CATEGORICAL_IMPUTER_LABELS[categorical_imputer_label]

        default_onehot = st.session_state.get("preprocessing_config", {}).get(
            "onehot_drop", DEFAULT_PREPROCESSING_CONFIG["onehot_drop"]
        )
        onehot_label = st.selectbox(
            "Кодування категоріальних ознак",
            options=list(ONEHOT_ENCODING_LABELS.keys()),
            index=list(ONEHOT_ENCODING_LABELS.keys()).index(
                ONEHOT_ENCODING_LABELS_REVERSE.get(default_onehot, "Скорочене кодування для бінарних ознак")
            ),
        )
        onehot_drop = ONEHOT_ENCODING_LABELS[onehot_label]

    feature_cols = [col for col in all_cols if col != target_col]
    numeric_preview = [col for col in feature_cols if col in numeric_columns(df_current)]
    categorical_preview = [col for col in feature_cols if col in categorical_columns(df_current)]

    with st.expander("Показати деталі попередньої обробки", expanded=False):
        st.write(
            "Pipeline fit виконується лише на train. Для числових ознак застосовуються стандартизація або нормалізація, "
            "а для категоріальних — One-Hot Encoding."
        )
        st.write(
            "Для обох стратегій використовується один і той самий train/test split. Межі Q1 і Q3 обчислюються лише "
            "на навчальній вибірці, тому тестові дані не впливають на категоризацію."
        )
        details_df = pd.DataFrame(
            {
                "Параметр": [
                    "Масштабування",
                    "Обробка пропущених числових ознак",
                    "Обробка пропущених категоріальних ознак",
                    "One-Hot Encoding",
                    "Test size",
                    "Random state",
                ],
                "Поточне значення": [
                    scaler_type,
                    numeric_imputer_label,
                    categorical_imputer_label,
                    onehot_label,
                    f"{test_size:.2f}",
                    str(int(random_state)),
                ],
            }
        )
        st.dataframe(details_df, use_container_width=True, hide_index=True)

    with st.expander("Список необроблених ознак і їх типи", expanded=False):
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.write("**Усі необроблені ознаки**")
            st.write(feature_cols if feature_cols else "—")
        with col_f2:
            st.write("**Розподіл за типами**")
            st.write({"numeric": numeric_preview, "categorical": categorical_preview})

    return SplitConfig(
        target_col=target_col,
        strategy_type=strategy_type,
        scaler_type=scaler_type,
        numeric_imputer_strategy=numeric_imputer_strategy,
        categorical_imputer_strategy=categorical_imputer_strategy,
        onehot_drop=onehot_drop,
        test_size=float(test_size),
        random_state=int(random_state),
    )


def render_split_actions(df_current: pd.DataFrame, config: SplitConfig | None) -> None:
    if config is None:
        return

    st.markdown("### 5. Формування навчальної та тестової вибірки")
    st.caption(
        "Після підтвердження конфігурації буде побудовано preprocessing pipeline і створено вибірки для навчання та оцінювання."
    )

    c1, c2 = st.columns([1.3, 0.8])
    with c1:
        if st.button("Підготувати ознаки й розподілити дані на вибірки", type="primary", use_container_width=True):
            try:
                artifacts = build_split_artifacts(df_current, config)
                save_split_to_session(artifacts)
                st.success(f"Дані підготовлено для стратегії: {STRATEGY_LABELS[config.strategy_type]}.")
            except Exception as exc:
                st.error(f"Помилка під час формування вибірок і preprocessing: {exc}")

    with c2:
        if st.button("Скинути розбиття на вибірки", use_container_width=True):
            invalidate_split_state()
            st.success("Train/test split, preprocessing pipeline і похідні стани очищено.")


def render_split_summary() -> None:
    if not st.session_state.get("split_ready", False):
        return

    st.markdown("### 6. Поточний стан даних")
    st.caption("Підсумок сформованої конфігурації, яка буде використана на наступних етапах моделювання.")

    strategy_label = STRATEGY_LABELS.get(st.session_state.get("strategy_type", "—"), "—")
    n_features = len(st.session_state.get("feature_names_after_preprocessing", []))
    train_shape = f"{st.session_state.X_train.shape[0]} × {st.session_state.X_train.shape[1]}"
    test_shape = f"{st.session_state.X_test.shape[0]} × {st.session_state.X_test.shape[1]}"
    target_name = st.session_state.get("target_col", "—")

    st.markdown(f"**Стратегія:** {strategy_label}")
    st.markdown(f"**Цільова змінна:** {target_name}")
    c1, c2 = st.columns(2)
    c1.metric("Навчальна вибірка", train_shape)
    c2.metric("Тестова вибірка", test_shape)

    with st.expander("Показати деталі попередньої обробки", expanded=False):
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Q1", f"{st.session_state.get('Q1', 0.0):.4f}")
        p2.metric("Q3", f"{st.session_state.get('Q3', 0.0):.4f}")
        p3.metric("Ознак після preprocessing", str(n_features))
        p4.metric("Спільний split", "так")

        config = st.session_state.get("preprocessing_config", {})
        if config:
            st.info(
                "Preprocessing pipeline fit виконується лише на train-вибірці. Обидві стратегії використовують спільний split "
                "і спільні пороги Q1/Q3, що зменшує ризик data leakage та забезпечує коректне порівняння."
            )
            st.json(config)

    summary_tab1, summary_tab2, summary_tab3, summary_tab4 = st.tabs(
        [
            "Порівняння raw vs processed",
            "Target preview",
            "Класи після дискретизації",
            "Назви ознак після preprocessing",
        ]
    )

    with summary_tab1:
        left, right = st.columns(2)
        with left:
            st.markdown("**X_train_raw (head)**")
            st.dataframe(st.session_state.X_train_raw.head(), use_container_width=True)
        with right:
            st.markdown("**X_train після pipeline (head)**")
            st.dataframe(st.session_state.X_train.head(), use_container_width=True)

    with summary_tab2:
        y_preview = pd.DataFrame(
            {
                "y_train": st.session_state.y_train.head(10),
                "y_train_cont": st.session_state.y_train_cont.head(10),
            }
        )
        st.dataframe(y_preview, use_container_width=True)

    with summary_tab3:
        cls_preview = pd.DataFrame(
            {
                "y_train_cls": st.session_state.y_train_cls.head(10),
                "y_test_cls": st.session_state.y_test_cls.head(10),
            }
        )
        st.dataframe(cls_preview, use_container_width=True)

        dist_col1, dist_col2 = st.columns(2)
        with dist_col1:
            st.markdown("**Розподіл класів у train**")
            st.dataframe(
                st.session_state.y_train_cls.value_counts().sort_index().rename_axis("class").reset_index(name="count"),
                use_container_width=True,
                hide_index=True,
            )
        with dist_col2:
            st.markdown("**Розподіл класів у test**")
            st.dataframe(
                st.session_state.y_test_cls.value_counts().sort_index().rename_axis("class").reset_index(name="count"),
                use_container_width=True,
                hide_index=True,
            )

    with summary_tab4:
        feature_df = pd.DataFrame({"feature_name": st.session_state.get("feature_names_after_preprocessing", [])})
        st.dataframe(feature_df, use_container_width=True, hide_index=True)

    st.success("Попередню обробку даних налаштовано, можна переходити до навчання моделей першого рівня.")
