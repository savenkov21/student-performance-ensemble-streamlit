import streamlit as st

from module_1 import page_analysis
from module_2 import page_preprocessing
from module_3 import page_visualization
from module_4 import page_training_first_level_models
from module_5 import page_formation_metafeatures
from module_6 import page_training_second_level_model
from module_7 import page_evaluation_results
from module_8 import page_strategy_comparison
# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="Модель прогнозування успішності студентів",
    page_icon="🔎",
    layout="centered"  # or "wide"
)
st.title("🔎 Модель прогнозування успішності студентів")

# ----------------------------
# Sidebar: stage selection
# ----------------------------
with st.sidebar:
    st.header("Навігація")
    page = st.radio("Виберіть етап:", 
        options=["Завантаження та первинний аналіз даних", "Попередня обробка даних та формування вибірок", 
                 "Візуалізація", "Навчання базових моделей першого рівня", 
                 "Формування метаознак", "Навчання моделей другого рівня", "Оцінювання якості моделей",
                 "Порівняння стратегій прогнозування"],  
        index=0)
    
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
    
elif page == "Порівняння стратегій прогнозування":
    page_strategy_comparison()