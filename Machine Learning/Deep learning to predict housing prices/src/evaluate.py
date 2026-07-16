
import numpy as np

from sklearn.metrics import (

    mean_squared_error,

    mean_absolute_error,

    r2_score

)


def evaluate(model, X_test, y_test):

    predictions = model.predict(X_test, verbose=0)

    mse = mean_squared_error(y_test, predictions)

    rmse = np.sqrt(mse)

    mae = mean_absolute_error(y_test, predictions)

    r2 = r2_score(y_test, predictions)

    print("=" * 50)

    print("Evaluation")

    print("=" * 50)

    print(f"MSE  : {mse:,.2f}")

    print(f"RMSE : {rmse:,.2f}")

    print(f"MAE  : {mae:,.2f}")

    print(f"R²   : {r2:.4f}")

    print("=" * 50)

    return {

        "MSE": mse,

        "RMSE": rmse,

        "MAE": mae,

        "R2": r2

    }
