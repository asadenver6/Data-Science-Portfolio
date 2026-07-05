import pandas as pd

def load_data(path: str):
    df = pd.read_csv(path)

    df = df[
        ['longitude','latitude','housing_median_age','total_rooms',
         'total_bedrooms','population','households',
         'median_income','ocean_proximity','median_house_value']
    ]

    X = df.drop("median_house_value", axis=1)
    y = df["median_house_value"]

    return X, y
