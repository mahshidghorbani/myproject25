import os
import time

from tensorflow.keras.callbacks import (
    TensorBoard,
    ModelCheckpoint,
    EarlyStopping,
    CSVLogger,
)

from data import Dataset
from models import ResearchModels


def train(batch_size=4, nb_epoch=10):
    os.makedirs(os.path.join("data", "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join("data", "logs", "lstm"), exist_ok=True)

    data = Dataset(seq_length=26, class_limit=2)

    X_train, y_train = data.get_all_sequences_in_memory("training")
    X_test, y_test = data.get_all_sequences_in_memory("testing")

    if len(X_train) == 0:
        raise ValueError("No training samples were loaded.")

    if len(X_test) == 0:
        raise ValueError("No testing samples were loaded.")

    print("Training data shape:", X_train.shape)
    print("Training labels shape:", y_train.shape)
    print("Testing data shape:", X_test.shape)
    print("Testing labels shape:", y_test.shape)
    print("Classes:", data.classes)

    rm = ResearchModels(
        nb_classes=len(data.classes),
        model="lstm",
        seq_length=data.seq_length,
    )

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    checkpoint_path = os.path.join(
        "data",
        "checkpoints",
        f"lstm-{timestamp}.keras",
    )

    callbacks = [
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        TensorBoard(
            log_dir=os.path.join("data", "logs", "lstm", timestamp)
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        CSVLogger(
            os.path.join("data", "logs", f"training-{timestamp}.csv")
        ),
    ]

    print("Starting model training...")

    rm.model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=nb_epoch,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1,
    )

    print("Evaluating model...")

    loss, accuracy = rm.model.evaluate(
        X_test,
        y_test,
        verbose=1,
    )

    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    print("Training finished.")


def main():
    train(batch_size=4, nb_epoch=10)


if __name__ == "__main__":
    main()