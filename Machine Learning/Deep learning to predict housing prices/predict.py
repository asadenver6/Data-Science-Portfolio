
import argparse
import pandas as pd
import joblib

from tensorflow.keras.models import load_model

from src.config import MODEL_PATH, SCALER_PATH


def load_artifacts():

    model = load_model(MODEL_PATH)

    scaler = joblib.load(SCALER_PATH)

    return model, scaler


def predict(input_file):

    model, scaler = load_artifacts()

    df = pd.read_csv(input_file)

    X = scaler.transform(df)

    predictions = model.predict(
        X,
        verbose=0
    )

    return predictions.flatten()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV file"
    )

    args = parser.parse_args()

    result = predict(args.input)

    print("\nPredictions:")
    
    for value in result:
        print(f"${value:,.2f}")
