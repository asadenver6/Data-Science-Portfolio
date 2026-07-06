
from pipelines.preprocessing import build_preprocessor

def test_preprocessing_runs(split_data):
    X_train, X_test, y_train, y_test = split_data

    preprocessor = build_preprocessor(X_train)
    Xt = preprocessor.fit_transform(X_train)

    assert Xt.shape[0] == X_train.shape[0]
    assert Xt is not None
