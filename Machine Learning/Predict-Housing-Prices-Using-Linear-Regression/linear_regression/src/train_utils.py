
from sklearn.pipeline import Pipeline
from pipelines.preprocessing import build_preprocessor
from src.evaluate import evaluate


def build_pipeline(model):
    return Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("model", model)
    ])


def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    results = {}
    trained_models = {}

    for name, model in models.items():

        pipeline = build_pipeline(model)

        pipeline.fit(X_train, y_train)

        metrics = evaluate(pipeline, X_test, y_test)

        results[name] = metrics
        trained_models[name] = pipeline

        print(f"\n===== {name.upper()} =====")
        print(metrics)

    return trained_models, results
