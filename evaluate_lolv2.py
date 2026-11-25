
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from models.adaptive_refinement import AdaptiveRefinement


# ==================================================
# CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "datasets" / "LOLv2_Real"
INPUT_DIR = DATASET_DIR / "Test" / "Input"
GT_DIR = DATASET_DIR / "Test" / "GT"

MODEL_PATH = (
    BASE_DIR / "checkpoints" / "best_resunet_exp5.keras"
)

OUTPUT_DIR = (
    BASE_DIR / "results" / "lolv2_cross_dataset"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_HEIGHT = 256
MODEL_WIDTH = 384
SATURATION_SCALE = 1.75


# ==================================================
# SATURATION ADJUSTMENT
# Same implementation as app.py
# ==================================================

def adjust_saturation(image, scale=SATURATION_SCALE):

    image = np.asarray(image, dtype=np.float32)
    image = np.clip(image, 0, 1)

    image_uint8 = np.round(
        image * 255
    ).astype(np.uint8)

    hsv = cv2.cvtColor(
        image_uint8,
        cv2.COLOR_RGB2HSV
    )

    hsv[:, :, 1] = np.clip(
        hsv[:, :, 1].astype(np.float32) * scale,
        0,
        255
    ).astype(np.uint8)

    result = cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2RGB
    )

    return result.astype(np.float32) / 255.0


# ==================================================
# METRICS
# ==================================================

def calculate_metrics(reference, prediction):

    reference = np.clip(
        reference.astype(np.float32), 0, 1
    )

    prediction = np.clip(
        prediction.astype(np.float32), 0, 1
    )

    reference_tensor = tf.convert_to_tensor(
        reference[np.newaxis, ...],
        dtype=tf.float32
    )

    prediction_tensor = tf.convert_to_tensor(
        prediction[np.newaxis, ...],
        dtype=tf.float32
    )

    psnr = tf.image.psnr(
        reference_tensor,
        prediction_tensor,
        max_val=1.0
    ).numpy()[0]

    ssim = tf.image.ssim(
        reference_tensor,
        prediction_tensor,
        max_val=1.0
    ).numpy()[0]

    mae = np.mean(
        np.abs(reference - prediction)
    )

    return float(psnr), float(ssim), float(mae)


# ==================================================
# LOAD MODEL
# ==================================================

def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    print("\nLoading Experiment 5 checkpoint...")

    model = tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False
    )

    print("Model loaded.")
    print("Parameters:", model.count_params())

    return model


# ==================================================
# MAIN EVALUATION
# ==================================================

def main():

    print("\n========================================")
    print(" NIGHTWATCH LOLv2 REAL EVALUATION")
    print("========================================")

    if not INPUT_DIR.exists() or not GT_DIR.exists():
        raise FileNotFoundError(
            "LOLv2 Input or GT directory not found."
        )

    low_paths = sorted(
        p for p in INPUT_DIR.iterdir()
        if p.is_file() and p.suffix.lower() == ".png"
    )

    gt_paths = sorted(
        p for p in GT_DIR.iterdir()
        if p.is_file() and p.suffix.lower() == ".png"
    )

    low_names = {p.name for p in low_paths}
    gt_names = {p.name for p in gt_paths}

    if low_names != gt_names:
        raise ValueError(
            "Input and GT filenames do not match.\n"
            f"Missing GT: {sorted(low_names - gt_names)}\n"
            f"Missing Input: {sorted(gt_names - low_names)}"
        )

    print("Matched test pairs:", len(low_paths))

    model = load_model()
    refiner = AdaptiveRefinement()

    stages = [
        "Original_Input",
        "Experiment_5",
        "Adaptive_Gamma_CLAHE",
        "Full_NightWatch"
    ]

    all_results = []
    errors = []

    for index, low_path in enumerate(low_paths, start=1):

        try:

            gt_path = GT_DIR / low_path.name

            # Read original images
            low_bgr = cv2.imread(str(low_path))
            gt_bgr = cv2.imread(str(gt_path))

            if low_bgr is None or gt_bgr is None:
                raise ValueError("Could not read input or GT image.")

            low_rgb = cv2.cvtColor(
                low_bgr, cv2.COLOR_BGR2RGB
            )

            gt_rgb = cv2.cvtColor(
                gt_bgr, cv2.COLOR_BGR2RGB
            )

            original_height, original_width = low_rgb.shape[:2]

            if gt_rgb.shape[:2] != low_rgb.shape[:2]:
                raise ValueError(
                    f"Input/GT dimensions differ: "
                    f"{low_rgb.shape[:2]} vs {gt_rgb.shape[:2]}"
                )

            # Convert to [0,1]
            low_float = low_rgb.astype(
                np.float32
            ) / 255.0

            gt_float = gt_rgb.astype(
                np.float32
            ) / 255.0

            # Resize input exactly as app.py
            resized = cv2.resize(
                low_rgb,
                (MODEL_WIDTH, MODEL_HEIGHT),
                interpolation=cv2.INTER_AREA
            )

            normalized = (
                resized.astype(np.float32) / 255.0
            )

            model_input = normalized[np.newaxis, ...]

            # Stage 1: Experiment 5
            prediction = model(
                model_input,
                training=False
            ).numpy()[0]

            prediction = np.clip(
                prediction, 0, 1
            )

            # Stage 2: Adaptive gamma + CLAHE
            adaptive = refiner.enhance(prediction)

            # Stage 3: Saturation adjustment
            final = adjust_saturation(
                adaptive,
                SATURATION_SCALE
            )

            # Restore original dimensions as app.py does
            model_output = cv2.resize(
                prediction,
                (original_width, original_height),
                interpolation=cv2.INTER_CUBIC
            )

            adaptive_output = cv2.resize(
                adaptive,
                (original_width, original_height),
                interpolation=cv2.INTER_CUBIC
            )

            final_output = cv2.resize(
                final,
                (original_width, original_height),
                interpolation=cv2.INTER_CUBIC
            )

            predictions = {
                "Original_Input": low_float,
                "Experiment_5": model_output,
                "Adaptive_Gamma_CLAHE": adaptive_output,
                "Full_NightWatch": final_output
            }

            # Calculate metrics against original-resolution GT
            for stage in stages:

                psnr, ssim, mae = calculate_metrics(
                    gt_float,
                    predictions[stage]
                )

                all_results.append({
                    "Image": low_path.name,
                    "Stage": stage,
                    "PSNR": psnr,
                    "SSIM": ssim,
                    "MAE": mae,
                    "Width": original_width,
                    "Height": original_height
                })

            print(
                f"Processed {index}/{len(low_paths)}: "
                f"{low_path.name}"
            )

        except Exception as error:

            print(
                f"ERROR processing {low_path.name}: {error}"
            )

            errors.append({
                "Image": low_path.name,
                "Error": str(error)
            })

    # --------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------

    if not all_results:
        raise RuntimeError(
            "No images were evaluated successfully."
        )

    results_df = pd.DataFrame(all_results)

    results_df.to_csv(
        OUTPUT_DIR / "lolv2_per_image_results.csv",
        index=False
    )

    summary_df = (
        results_df
        .groupby("Stage")
        .agg(
            Images_Evaluated=("Image", "count"),
            Mean_PSNR=("PSNR", "mean"),
            Std_PSNR=("PSNR", "std"),
            Mean_SSIM=("SSIM", "mean"),
            Std_SSIM=("SSIM", "std"),
            Mean_MAE=("MAE", "mean"),
            Std_MAE=("MAE", "std")
        )
        .reset_index()
    )

    summary_df.to_csv(
        OUTPUT_DIR / "lolv2_summary.csv",
        index=False
    )

    if errors:

        pd.DataFrame(errors).to_csv(
            OUTPUT_DIR / "evaluation_errors.csv",
            index=False
        )

    # --------------------------------------------------
    # IMPROVEMENT OVER EXPERIMENT 5
    # --------------------------------------------------

    pivot = results_df.pivot(
        index="Image",
        columns="Stage",
        values=["PSNR", "SSIM", "MAE"]
    )

    comparison_rows = []

    for stage in [
        "Adaptive_Gamma_CLAHE",
        "Full_NightWatch"
    ]:

        for metric in ["PSNR", "SSIM", "MAE"]:

            a = pivot[(metric, "Experiment_5")]
            b = pivot[(metric, stage)]

            if metric == "MAE":
                differences = a - b
            else:
                differences = b - a

            comparison_rows.append({
                "Stage": stage,
                "Metric": metric,
                "Mean_Improvement": differences.mean(),
                "Improved_Images": int(
                    (differences > 0).sum()
                ),
                "Unchanged_Images": int(
                    (differences == 0).sum()
                ),
                "Deteriorated_Images": int(
                    (differences < 0).sum()
                ),
                "Images_Compared": len(differences)
            })

    comparison_df = pd.DataFrame(comparison_rows)

    comparison_df.to_csv(
        OUTPUT_DIR / "lolv2_improvement.csv",
        index=False
    )

    # --------------------------------------------------
    # DISPLAY
    # --------------------------------------------------

    print("\n========================================")
    print(" LOLv2 REAL EVALUATION SUMMARY")
    print("========================================")

    print(
        summary_df.round(5).to_string(index=False)
    )

    print("\n========================================")
    print(" IMPROVEMENT OVER EXPERIMENT 5")
    print("========================================")

    print(
        comparison_df.round(5).to_string(index=False)
    )

    print("\nSuccessfully evaluated:",
          results_df["Image"].nunique(), "images")

    print("Failed images:", len(errors))

    print("\nResults saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
