import os
import numpy as np
import tensorflow as tf

# Dataset configuration
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(
    PROJECT_DIR, "LOLDataset"
)

IMAGE_SIZE = (256, 384)  # Height, Width
BATCH_SIZE = 8
SEED = 42

VALID_EXTENSIONS = (".png", ".jpg", ".jpeg")


def get_image_pairs(split):

    low_dir = os.path.join(DATASET_DIR, split, "low")
    high_dir = os.path.join(DATASET_DIR, split, "high")

    if not os.path.isdir(low_dir) or not os.path.isdir(high_dir):
        raise FileNotFoundError(f"Dataset folders missing: {split}")

    low_files = sorted(
        f for f in os.listdir(low_dir)
        if f.lower().endswith(VALID_EXTENSIONS)
    )

    high_files = sorted(
        f for f in os.listdir(high_dir)
        if f.lower().endswith(VALID_EXTENSIONS)
    )

    if set(low_files) != set(high_files):
        raise ValueError(f"Image pairing mismatch in {split}")

    low_paths = [
        os.path.join(low_dir, f) for f in low_files
    ]

    high_paths = [
        os.path.join(high_dir, f) for f in low_files
    ]

    return low_paths, high_paths


def load_image(low_path, high_path):

    # Read low-light image
    low_image = tf.io.read_file(low_path)
    low_image = tf.io.decode_image(
        low_image, channels=3, expand_animations=False
    )

    # Read high-light reference
    high_image = tf.io.read_file(high_path)
    high_image = tf.io.decode_image(
        high_image, channels=3, expand_animations=False
    )

    low_image.set_shape([None, None, 3])
    high_image.set_shape([None, None, 3])

    # Resize images
    low_image = tf.image.resize(
        low_image, IMAGE_SIZE
    )

    high_image = tf.image.resize(
        high_image, IMAGE_SIZE
    )

    # Normalize pixel values to [0, 1]
    low_image = tf.cast(low_image, tf.float32) / 255.0
    high_image = tf.cast(high_image, tf.float32) / 255.0

    return low_image, high_image


def create_pipeline(low_paths, high_paths, training=False):

    dataset = tf.data.Dataset.from_tensor_slices(
        (low_paths, high_paths)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(low_paths),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def build_datasets():

    # Load training and validation source pairs
    low_paths, high_paths = get_image_pairs("our485")

    # Reproducible 90:10 split
    indices = np.random.default_rng(SEED).permutation(
        len(low_paths)
    )

    split_index = int(len(indices) * 0.9)

    train_indices = indices[:split_index]
    val_indices = indices[split_index:]

    train_low = [low_paths[i] for i in train_indices]
    train_high = [high_paths[i] for i in train_indices]

    val_low = [low_paths[i] for i in val_indices]
    val_high = [high_paths[i] for i in val_indices]

    # Load independent evaluation dataset
    test_low, test_high = get_image_pairs("eval15")

    # Create TensorFlow pipelines
    train_ds = create_pipeline(
        train_low, train_high, training=True
    )

    val_ds = create_pipeline(
        val_low, val_high
    )

    test_ds = create_pipeline(
        test_low, test_high
    )

    print("NightWatch Data Pipeline")
    print("-----------------------")
    print("Training pairs:", len(train_low))
    print("Validation pairs:", len(val_low))
    print("Testing pairs:", len(test_low))
    print("Image size:", IMAGE_SIZE)
    print("Batch size:", BATCH_SIZE)

    return train_ds, val_ds, test_ds


if __name__ == "__main__":

    train_ds, val_ds, test_ds = build_datasets()

    for low, high in train_ds.take(1):
        print("\nSample batch verification")
        print("Low-light shape:", low.shape)
        print("High-light shape:", high.shape)
        print("Minimum pixel:", tf.reduce_min(low).numpy())
        print("Maximum pixel:", tf.reduce_max(low).numpy())
