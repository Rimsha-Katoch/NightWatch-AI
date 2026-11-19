import os
import tensorflow as tf

from data_loader import build_datasets
from models.unet import build_unet
from models.losses import combined_loss


# Project directories
PROJECT_DIR = "/content/NightWatch"

CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "checkpoints")
LOG_DIR = os.path.join(PROJECT_DIR, "logs")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Training configuration
EPOCHS = 5
LEARNING_RATE = 0.0001

print("Loading datasets...")

train_ds, val_ds, test_ds = build_datasets()

print("\nBuilding U-Net model...")

model = build_unet()

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss=combined_loss
)

# Save best model
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(
        CHECKPOINT_DIR, "best_nightwatch.keras"
    ),
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

# Stop when validation loss stops improving
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# Reduce learning rate when progress plateaus
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-7,
    verbose=1
)

# Save training history
csv_logger = tf.keras.callbacks.CSVLogger(
    os.path.join(LOG_DIR, "training_history.csv"),
    append=False
)

print("\nStarting NightWatch training...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr,
        csv_logger
    ]
)

# Save final model
model.save(
    os.path.join(CHECKPOINT_DIR, "final_nightwatch.keras")
)

print("\nTraining completed!")
print("Best model:", os.path.join(
    CHECKPOINT_DIR, "best_nightwatch.keras"
))
print("Training history:", os.path.join(
    LOG_DIR, "training_history.csv"
))
