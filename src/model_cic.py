from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    GlobalAveragePooling1D,
    Dense,
    Dropout,
    BatchNormalization,
    Reshape
)
from tensorflow.keras.optimizers import Adam


def build_cnn(input_shape, num_classes):

    model = Sequential([
        Input(shape=(input_shape,)),

        Reshape((input_shape, 1)),

        Conv1D(64, 3, activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling1D(2),

        Conv1D(128, 3, activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling1D(2),

        Conv1D(256, 3, activation="relu", padding="same"),
        BatchNormalization(),

        GlobalAveragePooling1D(),

        Dense(256, activation="relu"),
        Dropout(0.5),

        Dense(128, activation="relu"),
        Dropout(0.3),

        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    model = build_cnn(75, 5)
    model.summary()