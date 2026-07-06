
from src.model_factory import create_model
from pipelines.preprocessing import build_preprocessor
from src.model_factory import create_model

def test_model_fits(split_data):
    X_train, X_test, y_train, y_test = split_data

    preprocessor = build_preprocessor(X_train)
    model = create_model(preprocessor)

    sample_X = X_train.sample(200, random_state=42)
    sample_y = y_train.loc[sample_X.index]

    model.fit(sample_X, sample_y)

    preds = model.predict(sample_X)
    assert len(preds) == len(sample_X)
