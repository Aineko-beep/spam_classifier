import streamlit as st
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

model_display_names = [
    ("logreg", "Логистическая регрессия"),
    ("svm", "Метод опорных векторов (SVM)"),
    ("naive_bayes", "Multinomial Naive Bayes"),
    ("random_forest", "Случайный лес"),
    ("knn", "K-ближайших соседей (KNN)"),
]

model_choice = st.selectbox("Выберите модель:", model_display_names, format_func=lambda x: x[1])
model_name = model_choice[0]  # Берём техническое имя для загрузки .pkl

# Загрузка обученной модели и векторайзера
@st.cache_resource
def load_model_and_vectorizer(model_name):
    model = joblib.load(f"models/{model_name}.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    return model, vectorizer

# Интерфейс
st.title("Классификатор писем: Спам или не спам?")

model, vectorizer = load_model_and_vectorizer(model_name)

text_input = st.text_area("Введите текст письма:")

file_input = st.file_uploader("или загрузите текстовый файл", type=["txt"])

if file_input:
    text_input = file_input.read().decode("utf-8")

if st.button("Получить предсказание"):
    if text_input.strip() == "":
        st.warning("Пожалуйста, введите текст письма или загрузите файл.")
    else:
        text_vector = vectorizer.transform([text_input])
        prediction = model.predict(text_vector)[0]
        if prediction == 1:
            label = "СПАМ"
        else:
            label = "НЕ СПАМ"
        st.success(f"Предсказание: {label}")
