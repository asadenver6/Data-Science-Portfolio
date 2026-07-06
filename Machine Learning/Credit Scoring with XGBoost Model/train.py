
import joblib
from sklearn.model_selection import train_test_split

from src.data_loader import load_data
from pipelines.preprocessing import build_preprocessor
from src.model_factory import create_model
from src.train_utils import tune_model
from src.evaluate import evaluate_model

# 1. Load data
data = load_data("data/creditscoring.csv")

# 2. Split features/target
X = data.drop("good_bad", axis=1)
y = data["good_bad"].map({"bad": 0, "good": 1})

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. Build pipeline
preprocessor = build_preprocessor(X_train)
model = create_model(preprocessor)

# 5. Hyperparameter tuning
param_grid = {
    "model__max_depth": [3, 4, 5],
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__gamma": [0, 0.25, 1],
    "model__reg_lambda": [0, 1, 10],
    "model__scale_pos_weight": [1, 3, 5]
}

grid = tune_model(model, param_grid, X_train, y_train)

# 6. Evaluate best model
best_model = grid.best_estimator_
evaluate_model(best_model, X_test, y_test)

# 7. Save model
joblib.dump(best_model, "models/model.pkl")

print("Model saved to models/model.pkl")
