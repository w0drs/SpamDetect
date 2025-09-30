from ml.dataset.load_dataset import get_dataset
from contextlib import nullcontext as does_not_raise
from ml.utils.exceptions import CSVFileNotFoundError, CSVLoadingError
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

@pytest.mark.parametrize(
    argnames="folder_path, sep, encoding, expectation",
    argvalues=[
        (str(PROJECT_ROOT / "data" / "spam_ham_data.csv"), ",", "latin-1", does_not_raise()),
        (str(PROJECT_ROOT / "data" / "spam_ham.csv"), ",", "latin-1", pytest.raises(CSVFileNotFoundError)),
        (str(PROJECT_ROOT / "data" / "spam_ham_data.csv"), ",", "utf-8", pytest.raises(CSVLoadingError))
    ]
)
def test_scaler_load(folder_path: str, sep: str, encoding: str, expectation):
    with expectation:
        dataset = get_dataset(folder_path, sep=sep, encoding=encoding)
        assert dataset is not None

