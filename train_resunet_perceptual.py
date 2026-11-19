import os
import tensorflow as tf

from data_loader import build_datasets
from models.resunet import build_resunet
from models.advanced_losses import build_advanced_loss

BASE_DIR = "/content/NightWatch"

CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

EPOCHS = 30
BATCH_SIZE = 4
LEARNING_RATE = 0.0001


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

train_ds = train_ds.unbatch()

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

val_ds = val_ds.unbatch().batch(BATCH_SIZE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

print("Building Residual Attention U-Net...")

model = build_resunet(
    input_shape=(256, 384, 3)
)

print("Building VGG16 perceptual loss...")

advanced_loss, _ = build_advanced_loss()

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss=advanced_loss
)

model.summary()

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(
            CHECKPOINT_DIR,
            "best_resunet_perceptual.weights.h5"
        ),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
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
        os.path.join(
            LOG_DIR,
            "resunet_perceptual_history.csv"
        ),
        append=False
    )
]

print("\nStarting Experiment 4...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

model.save_weights(
    os.path.join(
        CHECKPOINT_DIR,
        "final_resunet_perceptual.weights.h5"
    )
)

print("\nExperiment 4 completed!")
