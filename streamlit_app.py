# streamlit_app.py
import streamlit as st
import sys
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Добавляем путь для импорта
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.model import MedicalTreatmentPlanner
except ImportError:
    from model import MedicalTreatmentPlanner

# Настройка страницы
st.set_page_config(
    page_title="Медицинский AI-ассистент",
    page_icon="🏥",
    layout="wide"
)

# Заголовок
st.title("🏥 Медицинский AI-ассистент")
st.markdown("---")

# Инициализация модели (с кэшированием)
@st.cache_resource
def load_planner():
    with st.spinner("Загрузка модели... Это может занять 1-2 минуты"):
        return MedicalTreatmentPlanner()

# Загружаем модель
try:
    planner = load_planner()
    st.sidebar.success("✅ Модель загружена")
except Exception as e:
    st.sidebar.error(f"❌ Ошибка загрузки модели: {e}")
    st.stop()

# Боковая панель с примерами
st.sidebar.header("📋 Примеры историй")

examples = {
    "Рак легкого": """Пациент 65 лет, центральный рак правого легкого (плоскоклеточный) IIIB ст.
Прогрессирование после 2 линий терапии. Иммунотерапия пембролизумабом - прогрессирование.
PD-L1 70%, ECOG 1. Требуется 3 линия терапии.""",
    
    "Рак молочной железы": """Пациентка 48 лет, рак молочной железы, люминальный B, HER2-негативный.
Метастазы в кости, печень. Проведено 5 линий ХТ. Прогрессирование.
Выявлена мутация PIK3CA. Статус ECOG 1.""",
    
    "Меланома": """Пациент 36 лет, меланома кожи, BRAF V600E мутация.
Метастазы в легкие, печень, головной мозг. Проведена иммунотерапия - прогрессирование.
Статус ECOG 1. Требуется следующая линия терапии."""
}

selected_example = st.sidebar.selectbox("Выберите пример:", ["", *examples.keys()])
if selected_example:
    default_text = examples[selected_example]
else:
    default_text = ""

# Основной интерфейс
col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 История болезни")
    history = st.text_area(
        "Введите историю болезни:",
        value=default_text,
        height=400,
        placeholder="Введите историю болезни здесь..."
    )
    
    col1_btn, col2_btn = st.columns(2)
    with col1_btn:
        generate_btn = st.button("🚀 Сгенерировать план", type="primary", use_container_width=True)
    with col2_btn:
        clear_btn = st.button("🗑️ Очистить", use_container_width=True)

with col2:
    st.subheader("📋 План лечения")
    output_placeholder = st.empty()
    
    if clear_btn:
        history = ""
        output_placeholder.markdown("*Здесь появится результат...*")
        st.rerun()

if generate_btn and history:
    with st.spinner("🔄 Анализ истории и генерация рекомендаций..."):
        try:
            result = planner.generate_with_citations(history)
            
            # Форматируем вывод
            output = f"""
## 📋 РЕКОМЕНДОВАННЫЙ ПЛАН ЛЕЧЕНИЯ
{result['plan']}

---

## 📚 ОБОСНОВАНИЯ
"""
            
            for i, citation in enumerate(result['citations'], 1):
                output += f"\n{i}. {citation.get('regimen', '')}\n"
                output += f"   📖 {citation.get('source', '')}\n"
                if citation.get('nccn'):
                    output += f"   🌐 {citation.get('nccn')}\n"
                if citation.get('evidence'):
                    output += f"   🔬 {citation.get('evidence')}\n"
            
            output_placeholder.markdown(output)
            
            # Техническая информация в expander
            with st.expander("🔬 Техническая информация"):
                st.json(result['extracted_info'])
                
        except Exception as e:
            st.error(f"❌ Ошибка: {str(e)}")

# Добавляем информацию в сайдбар
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📚 Источники
- Минздрав РФ (2024)
- NCCN Guidelines v.2.2024
- ESMO Practice Guidelines

### ⚠️ Дисклеймер
Система предназначена для помощи врачам и не заменяет клиническое решение.
""")