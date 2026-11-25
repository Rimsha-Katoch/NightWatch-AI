
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

MODEL_PATH = (
    BASE_DIR / "checkpoints" / "best_resunet_exp5.keras"
)

RESULT_DIR = (
    BASE_DIR / "results" / "adaptive_refinement_evaluation"
)

RESULT_DIR.mkdir(parents=True, exist_ok=True)

SATURATION_SCALE = 1.75


# ==================================================
# LOAD MODEL
# ==================================================

def load_model():

    print("\nLoading Experiment 5 checkpoint...")

    model = tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False
    )

    print("Model loaded successfully.")
    print("Parameters:", model.count_params())

    return model


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

    reference = tf.cast(reference, tf.float32)
    prediction = tf.cast(prediction, tf.float32)

    prediction = tf.clip_by_value(
        prediction, 0.0, 1.0
    )

    psnr = tf.image.psnr(
        reference, prediction, max_val=1.0
    )

    ssim = tf.image.ssim(
        reference, prediction, max_val=1.0
    )

    mae = tf.reduce_mean(
        tf.abs(reference - prediction),
        axis=[1, 2, 3]
    )

    return (
        psnr.numpy(),
        ssim.numpy(),
        mae.numpy()
    )


# ==================================================
# MAIN EVALUATION
# ==================================================

def main():

    print("\n========================================")
    print(" NIGHTWATCH ADAPTIVE REFINEMENT EVALUATION")
    print("========================================")

    # Load the same test pipeline used for
    # the six-model evaluation.

    from data_loader import build_datasets, get_image_pairs

    print("\nLoading LOL test dataset...")

    _, _, test_ds = build_datasets()

    low_paths, _ = get_image_pairs("eval15")
    filenames = [Path(p).name for p in low_paths]

    print("Test images:", len(filenames))

    # Load model and refinement module

    model = load_model()

    refiner = AdaptiveRefinement()

    all_results = []
    image_index = 0

    # --------------------------------------------------
    # Evaluate every test batch
    # --------------------------------------------------

    for low_images, high_images in test_ds:

        # Stage 1: Neural network prediction

        predictions = model(
            low_images,
            training=False
        )

        predictions = tf.clip_by_value(
            predictions, 0.0, 1.0
        )

        # Convert batch to NumPy for OpenCV processing

        low_np = low_images.numpy()
        high_np = high_images.numpy()
        predictions_np = predictions.numpy()

        # Stage 2: Adaptive gamma + CLAHE

        refined_batch = np.stack([
            refiner.enhance(image)
            for image in predictions_np
        ])

        # Stage 3: Saturation adjustment

        final_batch = np.stack([
            adjust_saturation(
                image,
                SATURATION_SCALE
            )
            for image in refined_batch
        ])

        # Calculate metrics for all four stages

        stages = {
            "Original_Input": low_np,
            "Experiment_5": predictions_np,
            "Adaptive_Gamma_CLAHE": refined_batch,
            "Full_NightWatch": final_batch
        }

        for stage_name, stage_images in stages.items():

            psnr, ssim, mae = calculate_metrics(
                high_np,
                stage_images
            )

            for i in range(len(stage_images)):

                if image_index + i >= len(filenames):
                    raise ValueError(
                        "More images evaluated than expected."
                    )

                all_results.append({
                    "Image": filenames[image_index + i],
                    "Stage": stage_name,
                    "PSNR": float(psnr[i]),
                    "SSIM": float(ssim[i]),
                    "MAE": float(mae[i])
                })

        image_index += len(low_np)

        print(
            f"Processed {image_index}/{len(filenames)} images"
        )

    # --------------------------------------------------
    # VALIDATION
    # --------------------------------------------------

    if image_index != len(filenames):
        raise ValueError(
            f"Expected {len(filenames)} images, "
            f"but processed {image_index}."
        )

    # --------------------------------------------------
    # SAVE PER-IMAGE RESULTS
    # --------------------------------------------------

    results_df = pd.DataFrame(all_results)

    per_image_path = (
        RESULT_DIR / "adaptive_per_image_results.csv"
    )

    results_df.to_csv(
        per_image_path,
        index=False
    )

    # --------------------------------------------------
    # SAVE SUMMARY
    # --------------------------------------------------

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

    summary_path = (
        RESULT_DIR / "adaptive_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    # --------------------------------------------------
    # IMPROVEMENT OVER EXPERIMENT 5
    # --------------------------------------------------

    baseline = summary_df[
        summary_df["Stage"] == "Experiment_5"
    ].iloc[0]

    comparison_rows = []

    for _, row in summary_df.iterrows():

        comparison_rows.append({
            "Stage": row["Stage"],
            "PSNR_Improvement": (
                row["Mean_PSNR"] - baseline["Mean_PSNR"]
            ),
            "SSIM_Improvement": (
                row["Mean_SSIM"] - baseline["Mean_SSIM"]
            ),
            "MAE_Reduction": (
                baseline["Mean_MAE"] - row["Mean_MAE"]
            )
        })

    comparison_df = pd.DataFrame(comparison_rows)

    comparison_df.to_csv(
        RESULT_DIR / "improvement_over_experiment5.csv",
        index=False
    )

    # --------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------

    print("\n========================================")
    print(" ADAPTIVE REFINEMENT RESULTS")
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

    print("\nEvaluation completed successfully.")

    print("\nResults saved in:")
    print(RESULT_DIR)

    print("\nGenerated files:")
    print("1. adaptive_per_image_results.csv")
    print("2. adaptive_summary.csv")
    print("3. improvement_over_experiment5.csv")


if __name__ == "__main__":
    main()
