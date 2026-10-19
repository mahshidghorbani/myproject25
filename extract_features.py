import csv
import glob
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import (
    InceptionV3,
    preprocess_input,
)
from tensorflow.keras.utils import load_img, img_to_array


# InceptionV3 extracts a 2048-dimensional feature vector
# for each image.
IMAGE_SIZE = (299, 299)
FEATURES_LENGTH = 2048
FEATURES_SUFFIX = 26

DATA_CSV = os.path.join("data", "data_file.csv")
SEQUENCES_DIR = os.path.join("data", "sequences")


def build_feature_extractor():
    """Create the pretrained InceptionV3 feature extraction model."""

    model = InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(299, 299, 3),
    )

    model.trainable = False
    return model


def extractor(image_path, model):
    """Extract a 2048-dimensional feature vector from one image."""

    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"Image file not found: {image_path}"
        )

    image = load_img(
        image_path,
        target_size=IMAGE_SIZE,
        color_mode="rgb",
    )

    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)

    features = model.predict(image_array, verbose=0)
    features = features[0]

    if features.shape != (FEATURES_LENGTH,):
        raise ValueError(
            f"Unexpected feature shape {features.shape} "
            f"for image: {image_path}"
        )

    return features.astype(np.float32)


def extract_features():
    """Extract and save feature sequences for the dataset videos."""

    if not os.path.isfile(DATA_CSV):
        raise FileNotFoundError(
            f"Dataset CSV not found: {DATA_CSV}. "
            "Run this script from the project root directory."
        )

    os.makedirs(SEQUENCES_DIR, exist_ok=True)

    print("Loading pretrained InceptionV3 model...")

    # Load the model only once, not once per image.
    feature_model = build_feature_extractor()

    print("InceptionV3 is ready.")

    with open(DATA_CSV, "r", newline="", encoding="utf-8-sig") as file:
        reader = csv.reader(file)

        for row_number, videos in enumerate(reader, start=1):
            # The original project expects at least three CSV columns.
            if not videos or all(not value.strip() for value in videos):
                continue

            if len(videos) < 3:
                raise ValueError(
                    f"CSV row {row_number} has fewer than 3 columns."
                )

            dataset_folder = videos[0].strip()
            video_folder = videos[1].strip()
            filename = videos[2].strip()

            if not filename:
                raise ValueError(
                    f"Missing video name in CSV row {row_number}."
                )

            path_frames = os.path.join(
                "data",
                dataset_folder,
                video_folder,
            )

            frame_pattern = os.path.join(
                path_frames,
                filename,
                "*.jpg",
            )

            frames = sorted(glob.glob(frame_pattern))

            # Also support uppercase JPG and JPEG extensions.
            frames += sorted(
                glob.glob(
                    os.path.join(
                        path_frames,
                        filename,
                        "*.JPG",
                    )
                )
            )
            frames += sorted(
                glob.glob(
                    os.path.join(
                        path_frames,
                        filename,
                        "*.jpeg",
                    )
                )
            )

            # Remove duplicate paths and keep a deterministic order.
            frames = sorted(set(frames))
            if not frames:
                raise FileNotFoundError(
                    f"No image frames found for video: {filename}"
                )

            output_path = os.path.join(
                SEQUENCES_DIR,
                f"{filename}-{FEATURES_SUFFIX}-features.npy",
            )

            sequence = []

            for frame_index, frame_path in enumerate(frames, start=1):
                features = extractor(frame_path, feature_model)
                sequence.append(features)
                print(
                    f"Video {filename}: "
                    f"processed frame {frame_index}/{len(frames)}"
                )

            feature_sequence = np.stack(sequence).astype(np.float32)
            np.save(output_path, feature_sequence)
            print(f"Saved features: {output_path}")

    print("Feature extraction completed successfully.")


if __name__ == "__main__":
    extract_features()