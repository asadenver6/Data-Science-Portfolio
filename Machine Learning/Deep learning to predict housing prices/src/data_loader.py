
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import (
    DATA_PATH,
    DROP_COLUMNS,
    TARGET,
    TEST_SIZE,
    VAL_SIZE,
    RANDOM_STATE
)


def load_data():

    df = pd.read_csv(DATA_PATH, skipinitialspace=True)

    df = df.drop(columns=DROP_COLUMNS)

    return df


def split_data(df):

    X = df.drop(columns=TARGET)

    y = df[TARGET]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=TEST_SIZE + VAL_SIZE,
        random_state=RANDOM_STATE
    )

    relative_val_size = VAL_SIZE / (TEST_SIZE + VAL_SIZE)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=1 - relative_val_size,
        random_state=RANDOM_STATE
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )
