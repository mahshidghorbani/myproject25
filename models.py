from collections import deque

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam


class ResearchModels:
    """Build and compile the LSTM model for driver drowsiness detection."""

    def __init__(
        self,
        nb_classes,
        model,
        seq_length,
        saved_model=None,
        features_length=2048,
    ):
        # Validate the model configuration.
        if nb_classes < 2:
            raise ValueError("nb_classes must be at least 2.")

        if seq_length < 1:
            raise ValueError("seq_length must be greater than 0.")

        if features_length < 1:
            raise ValueError("features_length must be greater than 0.")

        self.seq_length = seq_length
        self.saved_model = saved_model
        self.nb_classes = nb_classes
        self.features_length = features_length
        self.feature_queue = deque()

        # Input shape: (time steps, features per time step)
        self.input_shape = (seq_length, features_length)

        print("Building LSTM model...")

        # Build the model.
        self.model = self.lstm()

        # Compile the model.
        optimizer = Adam(learning_rate=0.00005)

        self.model.compile(
            loss="categorical_crossentropy",
            optimizer=optimizer,
            metrics=["accuracy"],
        )

        self.model.summary()

    def lstm(self):
        """Build an LSTM network for sequential CNN features."""
        model = keras.Sequential(
            [
                keras.Input(shape=self.input_shape),
                layers.LSTM(
                    128,
                    return_sequences=False,
                    dropout=0.3,
                ),
                layers.Dense(64, activation="relu"),
                layers.Dropout(0.3),
                layers.Dense(
                    self.nb_classes,
                    activation="softmax",
                ),
            ],
            name="drowsy_driver_lstm",
        )

        return model