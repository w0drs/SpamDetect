import joblib
import pandas as pd
from numpy import ndarray
from xgboost import XGBClassifier

from ml.utils.constants import MODEL_PARAMETERS


def train_model(x_train: pd.Series | ndarray, y_train: pd.Series | ndarray):
    try:
        model = XGBClassifier(
            **MODEL_PARAMETERS
        )
        model.fit(x_train, y_train)
        joblib.dump(model, "../models/xgboost_clf.pkl")
        return model
    except Exception as e:
        raise