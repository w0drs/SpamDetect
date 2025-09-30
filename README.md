# SpamDetector  
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
**Система обнаружения спам-сообщений на основе машинного обучения**  
Веб-приложение для классификации текстовых (английских) сообщений на спам/не спам.

## Что включает в себя проект?
- **ML-модель**: Классификация текста с помощью XGBoost + Sentence Transformers
- **REST API**: FastAPI
- **Веб-интерфейс**: Streamlit для удобного взаимодействия

## Архитектура
- Текст → Эмбеддинги: Sentence Transformer (BERT)
- Классификация: XGBoost
- Метрики: Accuracy, Precision, Recall, F1-score

## Метрики модели
- *Accuracy*: 0.98
- *Precision*: 0.95
- *Recall*: 0.87
- *f1-score*: 0.91

## Структура проекта
```text
SpamDetect/
├── backend/                    
│   ├── app.py                 # Основной веб-интерфейс
│   └── service.py             # FastApi приложение
├── data/                  
│   └── spam_ham_data.csv      # Исходные данные
├── ml/
│   ├── dataset/
|   |   ├── load_dataset.py    # Загрузчик датасета
|   |   ├── preprocessing.py   # Обработка текста
|   |   └── split_data.py      # Разделенме датасета
│   ├── training/
|   |   ├── model_training.py  # Обучение модели
|   |   └── prediction.py      # Прогнозы модели
│   ├── utils/
|   |   ├── constants.py       # Константные значения
|   |   └── exceptions.py      # Ошибки
|   └── train.py               # pipeline обучения модели
├── models/
|   └── xgboost_clf.pkl        # Обученная модель
├── notebooks/                 # Ноутбуки с экспериментами
|   ├── 1_EDA.ipynb
|   └── 2_Training.ipynb
├── tests/                     # Тесты
|   ├── test_dataset.py
|   ├── test_preprocessing.py
|   └── test_train.py
├── pytest.ini
└── README.md
```

## Локальный запуск
```bash
# API сервер
uvicorn backend.service:app --reload --port 8000

# Веб-интерфейс
streamlit run backend/app.py
```










