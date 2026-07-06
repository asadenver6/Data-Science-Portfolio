
from sklearn.model_selection import GridSearchCV

def tune_model(model, param_grid, X_train, y_train):
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring="roc_auc",
        verbose=2
    )

    grid.fit(X_train, y_train)
    return grid
