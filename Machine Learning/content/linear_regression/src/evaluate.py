
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)

    return {
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }
