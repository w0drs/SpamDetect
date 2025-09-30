import joblib
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from ml.utils.constants import EMBEDDING_MODEL_NAME, REVERSE_TARGET_MAPPING
from pydantic import BaseModel

class TextRequest(BaseModel):
    text: str

app = FastAPI()
xgboost_model = joblib.load("models/xgboost_clf.pkl")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

@app.post("/predict")
def predict(sentence: TextRequest):
    features = [
        sentence.text
    ]
    embeddings = embedding_model.encode(features)
    prediction = xgboost_model.predict(embeddings)
    return {"prediction": REVERSE_TARGET_MAPPING[int(prediction)]}