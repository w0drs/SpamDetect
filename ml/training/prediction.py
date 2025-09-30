import pandas as pd
from numpy import ndarray

def predict_values(model, x_test: pd.Series | ndarray):
    try:
        predictions = model.predict(x_test)
        return predictions
    except Exception as e:
        raise