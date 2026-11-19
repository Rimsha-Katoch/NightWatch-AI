import os
import tensorflow as tf

from data_loader import build_datasets
from models.resunet import build_resunet
from models.losses import combined_loss

EPOCHS = 30
BATCH_SIZE = 4
LEARNING_RATE = 0.0001

BASE_DIR = "/content/NightWatch"
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

print("Loading datasets...")

train_ds, val_ds, test_ds = build_datasets()

train_ds = train_ds.unbatch().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.unbatch().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print("Building Residual Attention U-Net...")

model = build_resunet(input_shape=(256, 384, 3))

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=combined_loss
)

model.summary()

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(CHECKPOINT_DIR, "best_resunet.keras"),
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
        os.path.join(LOG_DIR, "resunet_history.csv"),
        append=False
    )
]

print("\nStarting Residual Attention U-Net training...\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

model.save(os.path.join(CHECKPOINT_DIR, "final_resunet.keras"))

print("\nTraining completed successfully!")
