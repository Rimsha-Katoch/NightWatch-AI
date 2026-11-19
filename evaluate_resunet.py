import os
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from PIL import Image

from data_loader import build_datasets

BASE_DIR = "/content/NightWatch"

MODEL_PATH = os.path.join(
    BASE_DIR, "checkpoints", "best_resunet.keras"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR, "results", "resunet_eval15"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading best Residual Attention U-Net...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Loading test dataset...")

train_ds, val_ds, test_ds = build_datasets()

low_dir = Path(BASE_DIR) / "dataset" / "LOL" / "lol_dataset" / "eval15" / "low"

image_extensions = {".png", ".jpg", ".jpeg"}

image_paths = sorted([
    p for p in low_dir.iterdir()
    if p.suffix.lower() in image_extensions
])

results = []

image_index = 0

print("\nEvaluating test images...\n")

for low_images, high_images in test_ds:

    predictions = model(low_images, training=False)

    predictions = tf.clip_by_value(
        predictions, 0.0, 1.0
    )

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

        output_image = (
            predictions[i].numpy() * 255
        ).astype(np.uint8)

        Image.fromarray(output_image).save(
            os.path.join(OUTPUT_DIR, filename)
        )

        results.append({
            "Image": filename,
            "Baseline_PSNR": float(baseline_psnr[i].numpy()),
            "Enhanced_PSNR": float(enhanced_psnr[i].numpy()),
            "Baseline_SSIM": float(baseline_ssim[i].numpy()),
            "Enhanced_SSIM": float(enhanced_ssim[i].numpy())
        })

        image_index += 1

assert image_index == len(image_paths), (
    f"Evaluated {image_index} images, "
    f"but found {len(image_paths)} test images."
)

df = pd.DataFrame(results)

df["PSNR_Improvement"] = (
    df["Enhanced_PSNR"] - df["Baseline_PSNR"]
)

df["SSIM_Improvement"] = (
    df["Enhanced_SSIM"] - df["Baseline_SSIM"]
)

csv_path = os.path.join(
    OUTPUT_DIR, "evaluation_metrics.csv"
)

df.to_csv(csv_path, index=False)

print("\nINDIVIDUAL IMAGE RESULTS")
print(df.round(4).to_string(index=False))

print("\nAVERAGE RESULTS")
print("Baseline PSNR:", round(df["Baseline_PSNR"].mean(), 4))
print("Enhanced PSNR:", round(df["Enhanced_PSNR"].mean(), 4))
print("Baseline SSIM:", round(df["Baseline_SSIM"].mean(), 4))
print("Enhanced SSIM:", round(df["Enhanced_SSIM"].mean(), 4))

print("\nAVERAGE IMPROVEMENT")
print("PSNR:", round(df["PSNR_Improvement"].mean(), 4))
print("SSIM:", round(df["SSIM_Improvement"].mean(), 4))

print("\nEnhanced images saved in:", OUTPUT_DIR)
print("Metrics saved in:", csv_path)
