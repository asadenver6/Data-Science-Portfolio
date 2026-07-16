
import pickle

from tensorflow.keras.callbacks import (

    EarlyStopping,

    ModelCheckpoint,

    ReduceLROnPlateau

)

from src.config import (

    MODEL_PATH,

    HISTORY_PATH,

    BATCH_SIZE,

    EPOCHS

)


def train(

    model,

    X_train,

    y_train,

    X_val,

    y_val

):

    callbacks = [

        EarlyStopping(

            monitor="val_loss",

            patience=20,

            restore_best_weights=True

        ),

        ReduceLROnPlateau(

            monitor="val_loss",

            factor=0.5,

            patience=8,

            verbose=1

        ),

        ModelCheckpoint(

            MODEL_PATH,

            monitor="val_loss",

            save_best_only=True,

            verbose=1

        )

    ]

    history = model.fit(

        X_train,

        y_train,

        validation_data=(X_val, y_val),

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        callbacks=callbacks,

        verbose=1

    )

    with open(HISTORY_PATH, "wb") as f:

        pickle.dump(history.history, f)

    return history
