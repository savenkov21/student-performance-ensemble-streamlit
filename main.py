import streamlit as st

from app_pages.module_1 import page_analysis
from app_pages.module_2 import page_preprocessing
from app_pages.module_3 import page_visualization
from app_pages.module_4 import page_training_first_level_models
from app_pages.module_5 import page_formation_metafeatures
from app_pages.module_6 import page_training_second_level_model
from app_pages.module_7 import page_evaluation_results
from app_pages.module_8 import page_statistical_analysis
from app_pages.module_9 import page_strategy_comparison


st.set_page_config(
    page_title="Модель прогнозування успішності студентів",
    page_icon="🔎",
    layout="centered",
)

st.title("🔎 Модель прогнозування успішності студентів")

with st.sidebar:
    st.header("Навігація")
    page = st.radio(
        "Виберіть етап:",
        options=[
            "Завантаження та первинний аналіз даних",
            "Попередня обробка даних та формування вибірок",
            "Візуалізація",
            "Навчання базових моделей першого рівня",
            "Формування метаознак",
            "Навчання моделей другого рівня",
            "Оцінювання якості моделей",
            "Статистичний аналіз результатів класифікації",
            "Порівняння стратегій прогнозування",
        ],
        index=0,
    )
    st.markdown("---")
    st.caption("Порада: за потреби перемкніться на «широкий» макет у меню вгорі праворуч.")

if page == "Завантаження та первинний аналіз даних":
    page_analysis()
elif page == "Попередня обробка даних та формування вибірок":
    page_preprocessing()
elif page == "Візуалізація":
    page_visualization()
elif page == "Навчання базових моделей першого рівня":
    page_training_first_level_models()
elif page == "Формування метаознак":
    page_formation_metafeatures()
elif page == "Навчання моделей другого рівня":
    page_training_second_level_model()
elif page == "Оцінювання якості моделей":
    page_evaluation_results()
elif page == "Статистичний аналіз результатів класифікації":
    page_statistical_analysis()
elif page == "Порівняння стратегій прогнозування":
    page_strategy_comparison()
