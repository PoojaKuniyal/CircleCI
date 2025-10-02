import os

FILE_PATH = "artifacts/raw/data.csv"
PROCESSED_DATA_PATH = "artifacts/processed"

X_TRAIN_PROCESSED = os.path.join(PROCESSED_DATA_PATH, "X_train.pkl")

X_TEST_PROCESSED = os.path.join(PROCESSED_DATA_PATH, "X_test.pkl")

Y_TRAIN_PROCESSED = os.path.join(PROCESSED_DATA_PATH,"y_train.pkl")

Y_TEST_PROCESSED = os.path.join(PROCESSED_DATA_PATH,"y_test.pkl")

MODEL_PATH = "artifacts/models"

SAVE_MODEL_PATH = "artifacts/models/model.pkl"