
import os
import tensorflow as tf
import pandas as pd

from data_loader import build_datasets
from models.resunet import build_resunet
from models.experiment6_losses import build_experiment6_loss

# Reproducibility
tf.keras.utils.set_random_seed(42)

# Configuration
EPOCHS = 35
LEARNING_RATE = 1e-4

# GPU check
print("GPU:", tf.config.list_physical_devices("GPU"))

# Load datasets
train_ds, val_ds, test_ds = build_datasets()

# Paired horizontal flip augmentation
def augment(low, high):
    flip = tf.random.uniform(()) > 0.5
    low = tf.cond(flip, lambda: tf.image.flip_left_right(low),
                  lambda: low)
    high = tf.cond(flip, lambda: tf.image.flip_left_right(high),
                   lambda: high)
    return low, high

train_ds = train_ds.map(
    augment, num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

# Build model
model = build_resunet()

loss_fn, _ = build_experiment6_loss()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=loss_fn
)

# Separate Experiment 6 checkpoint paths
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)

best_path = "checkpoints/best_resunet_exp6.keras"
final_path = "checkpoints/final_resunet_exp6.keras"
log_path = "logs/resunet_exp6_history.csv"

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        best_path,
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
    tf.keras.callbacks.CSVLogger(log_path)
]

print("\nStarting Experiment 6 training...")
print("Loss weights: L1=0.45, SSIM=0.20, Brightness=0.15,")
print("              Contrast=0.10, Saturation=0.10")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Save final model
model.save(final_path)

print("\nExperiment 6 training completed.")
print("Best checkpoint:", best_path)
print("Final model:", final_path)
print("Training log:", log_path)
print("Model parameters:", model.count_params())
