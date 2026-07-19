from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D,
    MaxPooling1D,
    Dense,
    Dropout,
    Flatten,
    BatchNormalization,
    Reshape
)

def build_cnn(input_shape):

    model = Sequential()

    model.add(Reshape((input_shape,1), input_shape=(input_shape,)))

    model.add(Conv1D(64,3,activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))

    model.add(Conv1D(128,3,activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))

    model.add(Flatten())

    model.add(Dense(256,activation="relu"))
    model.add(Dropout(0.5))

    model.add(Dense(128,activation="relu"))
    model.add(Dropout(0.3))

    model.add(Dense(1,activation="sigmoid"))

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model