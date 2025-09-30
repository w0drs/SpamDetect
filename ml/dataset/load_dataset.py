import pandas as pd
from ml.utils.exceptions import CSVLoadingError, CSVFileNotFoundError

def get_dataset(file_path: str, encoding: str, sep: str):
    try:
        dataset = pd.read_csv(file_path, sep=sep, encoding=encoding)
    except FileNotFoundError:
        raise CSVFileNotFoundError(file_path)
    except UnicodeDecodeError as e:
        raise CSVLoadingError(file_path, f"Ошибка кодировки: {e}")
    except Exception as e:
        raise CSVLoadingError(file_path, str(e))
    return dataset