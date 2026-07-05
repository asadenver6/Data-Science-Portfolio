
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

def get_models():
    return {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=0.1),
        "lasso": Lasso(alpha=0.5, max_iter=10000),
        "elastic_net": ElasticNet()
    }
