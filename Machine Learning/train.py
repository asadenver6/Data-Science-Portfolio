
import joblib

from src.data_loader import load_data
from src.model_factory import get_models
from src.train_utils import train_and_evaluate

from sklearn.model_selection import train_test_split


def main():

    # Load data
    X, y = load_data('housing.csv')

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Models
    models = get_models()

    # Train
    trained_models, results = train_and_evaluate(
        models, X_train, X_test, y_train, y_test
    )

    # Save best model (example: ridge)
    best_model = trained_models["ridge"]

    joblib.dump(best_model, "models/model.pkl")

    print("\nModel saved to models/model.pkl")


if __name__ == "__main__":
    main()
