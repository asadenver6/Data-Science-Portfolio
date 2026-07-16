
import joblib

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import MinMaxScaler

from src.config import SCALER_PATH


class Preprocessor:

    def __init__(self):

        self.pipeline = Pipeline([
            ("scaler", MinMaxScaler())
        ])

    def fit_transform(self, X):

        X_scaled = self.pipeline.fit_transform(X)

        joblib.dump(self.pipeline, SCALER_PATH)

        return X_scaled

    def transform(self, X):

        return self.pipeline.transform(X)

    @staticmethod
    def load():

        return joblib.load(SCALER_PATH)
