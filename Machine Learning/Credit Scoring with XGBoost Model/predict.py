
import joblib
import pandas as pd

def load_model(path="models/model.pkl"):
    return joblib.load(path)

def predict(data_path):
    model = load_model()

    data = pd.read_csv(data_path)
    preds = model.predict(data)
    probs = model.predict_proba(data)[:, 1]

    return preds, probs


if __name__ == "__main__":
    preds, probs = predict("data/creditscoring.csv")
    print(preds[:10])
    print(probs[:10])
