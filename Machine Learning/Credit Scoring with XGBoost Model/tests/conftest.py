import pytest
from src.data_loader import load_data
from sklearn.model_selection import train_test_split

@pytest.fixture(scope="session")
def raw_data():
    return load_data("data/creditscoring.csv")


@pytest.fixture(scope="session")
def X_y(raw_data):
    X = raw_data.drop("good_bad", axis=1)
    y = raw_data["good_bad"].map({"bad": 0, "good": 1})
    return X, y


@pytest.fixture(scope="session")
def split_data(X_y):
    X, y = X_y
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
