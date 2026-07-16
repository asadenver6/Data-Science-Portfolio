
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from src.config import (
    INPUT_DIM,
    LEARNING_RATE
)


def build_model(
    hidden_layers=[32, 32],
    activation="relu"
):
    """
    Build and compile a neural network.

    Parameters
    ----------
    hidden_layers : list
        Number of neurons per hidden layer

    activation : str
        Activation function

    Returns
    -------
    keras.Model
    """

    model = Sequential()

    # First hidden layer
    model.add(
        Dense(
            hidden_layers[0],
            activation=activation,
            input_shape=(INPUT_DIM,)
        )
    )

    # Remaining hidden layers
    for units in hidden_layers[1:]:

        model.add(
            Dense(
                units,
                activation=activation
            )
        )

    # Output layer

    model.add(Dense(1))

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    )

    model.compile(

        optimizer=optimizer,

        loss="mse",

        metrics=[
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
            "mae"
        ]

    )

    return model
