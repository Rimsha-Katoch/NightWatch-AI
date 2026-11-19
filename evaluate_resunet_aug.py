import os
import tensorflow as tf
import numpy as np
import pandas as pd

from pathlib import Path
from PIL import Image
from data_loader import build_datasets

BASE_DIR = "/content/NightWatch"

MODEL_PATH = os.path.join(
    BASE_DIR, "checkpoints", "best_resunet_aug.keras"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR, "results", "resunet_aug_eval15"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading augmented Residual Attention U-Net...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Loading test dataset...")

_, _, test_ds = build_datasets()

low_dir = Path(
    BASE_DIR / Path("dataset/LOL/lol_dataset/eval15/low")
)

image_paths = sorted([
    p for p in low_dir.iterdir()
    if p.suffix.lower() in [".png", ".jpg", ".jpeg"]
])

results = []
image_index = 0

for low_images, high_images in test_ds:

    predictions = model(low_images, training=False)
    predictions = tf.clip_by_value(predictions, 0.0, 1.0)

    baseline_psnr = tf.image.psnr(
        high_images, low_images, max_val=1.0
    )

    enhanced_psnr = tf.image.psnr(
        high_images, predictions, max_val=1.0
    )

    baseline_ssim = tf.image.ssim(
        high_images, low_images, max_val=1.0
    )

    enhanced_ssim = tf.image.ssim(
        high_images, predictions, max_val=1.0
    )

    for i in range(low_images.shape[0]):

        filename = image_paths[image_index].name

        output = (
            predictions[i].numpy() * 255
        ).round().clip(0, 255).astype(np.uint8)

        Image.fromarray(output).save(
            os.path.join(OUTPUT_DIR, filename)
        )

        results.append({
            "Image": filename,
            "Baseline_PSNR": float(baseline_psnr[i]),
            "Enhanced_PSNR": float(enhanced_psnr[i]),
            "Baseline_SSIM": float(baseline_ssim[i]),
            "Enhanced_SSIM": float(enhanced_ssim[i])
        })

        image_index += 1

assert image_index == len(image_paths)

df = pd.DataFrame(results)

df["PSNR_Improvement"] = (
    df["Enhanced_PSNR"] - df["Baseline_PSNR"]
)

df["SSIM_Improvement"] = (
    df["Enhanced_SSIM"] - df["Baseline_SSIM"]
)

df.to_csv(
    os.path.join(OUTPUT_DIR, "evaluation_metrics.csv"),
    index=False
)

print("\nINDIVIDUAL RESULTS")
print(df.round(4).to_string(index=False))

print("\nAVERAGE RESULTS")
print(df.mean(numeric_only=True).round(4))

print("\nSaved enhanced images to:", OUTPUT_DIR)
