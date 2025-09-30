import nltk
import string
import joblib
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from ml.utils.constants import EMBEDDING_MODEL_NAME

nltk.download("punkt")
nltk.download('punkt_tab')
nltk.download('stopwords')

def tokenize_text(text: str, remove_stop_words: bool = True):
    """
    Токенизирование текста для tf-idf
    """
    try:
        snowball = SnowballStemmer(language="english")
        english_stop_words = stopwords.words("english")

        # Разделяем (токенизируем) отдельные элементы (буквы или символы) в тексте
        tokens = word_tokenize(text, language="english")
        # Убираем символы пунктуации
        tokens = [i.lower() for i in tokens if i not in string.punctuation]
        # Убираем стоп слова (дополнительные слова в языках, которые не несут смысловой нагрузки)
        if remove_stop_words:
            tokens = [i for i in tokens if i not in english_stop_words]
        # Приводим к нижнему регистру и удаляем окончания
        tokens = [snowball.stem(i) for i in tokens]

        return tokens
    except Exception as e:
        raise

def transform_text_with_tfidf(sentence: pd.DataFrame | pd.Series, fit_transform: bool = True):
    if fit_transform:
        try:
            vectorizer = TfidfVectorizer(
                tokenizer=lambda text: tokenize_text(text, remove_stop_words=True),
                token_pattern=None
            )
            features = vectorizer.fit_transform(sentence)
            joblib.dump(vectorizer, "../models/tfidf_vectorizer.pkl")
        except Exception as e:
            raise
        return features
    else:
        try:
            vectorizer = joblib.load("../models/tfidf_vectorizer.pkl")
            features = vectorizer.transform(sentence)
        except Exception as e:
            raise
        return features

def transform_text_with_embedding(sentence: pd.Series):
    try:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        embeddings = embedding_model.encode(sentence.tolist())
    except Exception as e:
        raise
    return embeddings
