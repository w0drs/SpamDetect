EMBEDDING_MODEL_NAME: str = 'all-MiniLM-L6-v2'
TARGET_MAPPING: dict = {"spam": 1, "ham": 0}
REVERSE_TARGET_MAPPING: dict = {1: "spam", 0: "ham"}
RANDOM_STATE: int = 52
TEST_SIZE: float = 0.2
MODEL_PARAMETERS: dict = {
    "scale_pos_weight": 6.5,
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "random_state": 42
}
