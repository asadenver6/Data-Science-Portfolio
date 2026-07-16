
from src.data_loader import (
    load_data,
    split_data
)

from src.preprocessing import Preprocessor

from src.model import build_model

from src.trainer import train

from src.evaluate import evaluate


def main():

    print("=" * 60)
    print("Housing Price Prediction")
    print("=" * 60)

    # --------------------------
    # Load data
    # --------------------------

    df = load_data()

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    # --------------------------
    # Preprocessing
    # --------------------------

    preprocessor = Preprocessor()

    X_train = preprocessor.fit_transform(X_train)

    X_val = preprocessor.transform(X_val)

    X_test = preprocessor.transform(X_test)

    # --------------------------
    # Model
    # --------------------------

    model = build_model(

        hidden_layers=[32,32,32]

    )

    model.summary()

    # --------------------------
    # Train
    # --------------------------

    train(

        model,

        X_train,

        y_train,

        X_val,

        y_val

    )

    # --------------------------
    # Evaluate
    # --------------------------

    evaluate(

        model,

        X_test,

        y_test

    )


if __name__ == "__main__":

    main()
