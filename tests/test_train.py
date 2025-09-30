from ml.dataset.load_dataset import get_dataset
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

def test_train_model():
    data = get_dataset(str(PROJECT_ROOT / "data" / "spam_ham_data.csv"), "latin-1", ",")