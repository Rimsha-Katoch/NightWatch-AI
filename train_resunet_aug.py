import os
import tensorflow as tf

from data_loader import build_datasets
from models.resunet import build_resunet
from models.losses import combined_loss

# Configuration
EPOCHS = 30
BATCH_SIZE = 4
LEARNING_RATE = 0.0001

BASE_DIR = "/content/NightWatch"
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# Paired horizontal flip augmentation
def augment_pair(low, high):

    flip = tf.random.uniform(()) > 0.5

    low = tf.cond(
        flip,
        lambda: tf.image.flip_left_right(low),
        lambda: low
    )

    high = tf.cond(
        flip,
        lambda: tf.image.flip_left_right(high),
        lambda: high
    )

    return low, high


print("Loading datasets...")

train_ds, val_ds, test_ds = build_datasets()

# Unbatch before augmentation
train_ds = train_ds.unbatch()

# Shuffle and augment paired images
train_ds = train_ds.shuffle(
    436,
    seed=42,
    reshuffle_each_iteration=True
)

train_ds = train_ds.map(
    augment_pair,
    num_parallel_calls=tf.data.AUTOTUNE
)

train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

# Validation remains unchanged
val_ds = val_ds.unbatch().batch(BATCH_SIZE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

print("Training examples:", 436)
print("Validation examples:", 49)
print("Test examples:", 15)

print("\nBuilding Residual Attention U-Net...")

model = build_resunet(
    input_shape=(256, 384, 3)
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss=combined_loss
)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(
            CHECKPOINT_DIR,
            "best_resunet_aug.keras"
        ),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    ),

    tf.keras.callbacks.CSVLogger(
        os.path.join(LOG_DIR, "resunet_aug_history.csv"),
        append=False
    )
]

print("\nStarting Experiment 3: Augmented Training")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

model.save(
    os.path.join(
        CHECKPOINT_DIR,
        "final_resunet_aug.keras"
    )
)

print("\nExperiment 3 completed!")
