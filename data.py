import csv
import os
import numpy as np
from tensorflow.keras.utils import to_categorical


class Dataset:
    def __init__(self, seq_length=26, class_limit=2, image_shape=(56, 24, 3)):
        self.seq_length = seq_length
        self.class_limit = class_limit
        self.image_shape = image_shape
        self.sequence_path = os.path.join("data", "sequences")
        self.data = self.get_data()
        self.classes = self.get_classes()

    def get_data(self):
        csv_path = os.path.join("data", "data_file.csv")
        with open(csv_path, "r", newline="", encoding="utf-8-sig") as fin:
            reader = csv.reader(fin)
            data = list(reader)
        return data

    def get_classes(self):
        classes = []
        for item in self.data:
            if len(item) >= 2 and item[1] not in classes:
                classes.append(item[1])
        classes = sorted(classes)
        if self.class_limit is not None:
            return classes[:self.class_limit]
        return classes

    def get_class_one_hot(self, class_str):
        label_encoded = self.classes.index(class_str)
        return to_categorical(label_encoded, num_classes=len(self.classes))

    def get_all_sequences_in_memory(self, train_test):
        print(f"Loading samples into memory for --> {train_test}")
        X, y = [], []
        for video in self.data:
            if len(video) < 3 or video[0] != train_test:
                continue
            sequence = self.get_extracted_sequence(video)
            if sequence is None:
                raise FileNotFoundError(
                    f"Feature sequence not found for video: {video[2]}"
                )
            X.append(sequence)
            y.append(self.get_class_one_hot(video[1]))
        return np.array(X), np.array(y)

    def get_extracted_sequence(self, video):
        filename = video[2]
        path = os.path.join(
            self.sequence_path,
            f"{filename}-{self.seq_length}-features.npy",
        )

        if not os.path.isfile(path):
            return None

        sequence = np.load(path)

        # Normalize feature shape to (sequence_length, 2048).
        sequence = sequence.reshape(sequence.shape[0], -1)

        if sequence.shape[0] != self.seq_length:
            raise ValueError(
                f"Invalid sequence length for {filename}: "
                f"expected {self.seq_length}, got {sequence.shape[0]}"
            )

        if sequence.shape[1] != 2048:
            raise ValueError(
                f"Invalid feature length for {filename}: "
                f"expected 2048, got {sequence.shape[1]}"
            )

        return sequence