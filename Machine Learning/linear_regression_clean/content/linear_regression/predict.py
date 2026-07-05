
import joblib
import pandas as pd


def load_model(path="models/model.pkl"):
    return joblib.load(path)


def predict(sample: dict):
    model = load_model()

    df = pd.DataFrame([sample])

    prediction = model.predict(df)

    return prediction[0]


if __name__ == "__main__":

    sample_input = {
        "longitude": -122.23,
        "latitude": 37.88,
        "housing_median_age": 41,
        "total_rooms": 880,
        "total_bedrooms": 129,
        "population": 322,
        "households": 126,
        "median_income": 8.3252,
        "ocean_proximity": "NEAR BAY"
    }

    pred = predict(sample_input)

    print("Predicted house value:", pred)
