
import os
import tensorflow as tf

from data_loader import build_datasets
from models.resunet import build_resunet
from models.experiment5_losses import build_experiment5_loss

# ==========================================
# EXPERIMENT 5 CONFIGURATION
# ==========================================

EPOCHS = 35
BATCH_SIZE = 4
LEARNING_RATE = 0.0001

BASE_DIR = "/content/NightWatch"

CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ==========================================
# PAIRED HORIZONTAL FLIP AUGMENTATION
# ==========================================

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


# ==========================================
# LOAD DATASETS
# ==========================================

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

print("Training examples: 436")
print("Validation examples: 49")
print("Testing examples: 15")

# ==========================================
# BUILD MODEL
# ==========================================

print("\nBuilding Residual Attention U-Net...")

model = build_resunet(
    input_shape=(256, 384, 3)
)

print("Model parameters:", model.count_params())

# ==========================================
# EXPERIMENT 5 LOSS
# ==========================================

experiment5_loss, component_fn = build_experiment5_loss()

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss=experiment5_loss
)

# ==========================================
# CALLBACKS
# ==========================================

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(
            CHECKPOINT_DIR,
            "best_resunet_exp5.keras"
        ),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),

    tf.keras.callbacks.CSVLogger(
        os.path.join(
            LOG_DIR,
            "resunet_exp5_history.csv"
        ),
        append=False
    )
]

# ==========================================
# TRAINING
# ==========================================

print("\nStarting Experiment 5 Training")
print("Epochs:", EPOCHS)
print("Batch size:", BATCH_SIZE)
print("Learning rate:", LEARNING_RATE)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ==========================================
# SAVE FINAL MODEL
# ==========================================

model.save(
    os.path.join(
        CHECKPOINT_DIR,
        "final_resunet_exp5.keras"
    )
)

print("\nExperiment 5 completed successfully!")
print("Best checkpoint:",
      os.path.join(CHECKPOINT_DIR, "best_resunet_exp5.keras"))
print("Final checkpoint:",
      os.path.join(CHECKPOINT_DIR, "final_resunet_exp5.keras"))
print("Training log:",
      os.path.join(LOG_DIR, "resunet_exp5_history.csv"))
