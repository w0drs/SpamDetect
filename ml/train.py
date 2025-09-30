from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from ml.dataset.load_dataset import get_dataset
from ml.dataset.preprocessing import transform_text_with_embedding
from ml.dataset.split_data import get_features
from ml.training.model_training import train_model
from ml.training.prediction import predict_values
from ml.utils.constants import RANDOM_STATE, TEST_SIZE


def main():
    data = get_dataset("../data/spam_ham_data.csv", "latin-1", ",")
    X, y = get_features(data)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    train_embeddings = transform_text_with_embedding(X_train)
    model = train_model(train_embeddings, y_train)
    print("обучение успешно выполнено!")

    test_embeddings = transform_text_with_embedding(X_test)
    y_pred = predict_values(model, test_embeddings)

    report = classification_report(y_test, y_pred)
    print(report)


if __name__ == "__main__":
    main()