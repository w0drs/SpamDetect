import pandas as pd

from ml.utils.constants import TARGET_MAPPING


def get_features(data: pd.DataFrame):
    X = data.v2
    y = data.v1

    return X, y.map(TARGET_MAPPING)

