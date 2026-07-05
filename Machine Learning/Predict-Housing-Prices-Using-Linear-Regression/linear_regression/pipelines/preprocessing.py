from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder


def build_preprocessor():
    numeric_features = [
        'longitude','latitude','housing_median_age','total_rooms',
        'total_bedrooms','population','households','median_income'
    ]

    categorical_features = ['ocean_proximity']

    numeric_transformer = Pipeline(steps=[
        ("imputer", IterativeImputer(random_state=42))
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return preprocessor
