
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from data_loader import build_datasets, get_image_pairs
from models.unet import build_unet
from models.resunet import build_resunet


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
RESULT_DIR = BASE_DIR / "results" / "six_model_evaluation"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# MODEL CONFIGURATIONS
# --------------------------------------------------

MODEL_CONFIGS = [
    {
        "name": "Experiment_1_UNet",
        "type": "full",
        "path": CHECKPOINT_DIR / "best_nightwatch.keras"
    },
    {
        "name": "Experiment_2_ResUNet",
        "type": "full",
        "path": CHECKPOINT_DIR / "best_resunet.keras"
    },
    {
        "name": "Experiment_3_Augmented",
        "type": "full",
        "path": CHECKPOINT_DIR / "best_resunet_aug.keras"
    },
    {
        "name": "Experiment_4_Perceptual",
        "type": "weights",
        "path": CHECKPOINT_DIR / "best_resunet_perceptual.weights.h5"
    },
    {
        "name": "Experiment_5",
        "type": "full",
        "path": CHECKPOINT_DIR / "best_resunet_exp5.keras"
    },
    {
        "name": "Experiment_6",
        "type": "full",
        "path": CHECKPOINT_DIR / "best_resunet_exp6.keras"
    }
]


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

def load_model(config):

    print(f"\nLoading {config['name']}...")

    if not config["path"].exists():
        raise FileNotFoundError(config["path"])

    if config["type"] == "weights":

        model = build_resunet(
            input_shape=(256, 384, 3)
        )

        model.load_weights(str(config["path"]))

    else:

        model = tf.keras.models.load_model(
            str(config["path"]),
            compile=False
        )

    print("Model loaded successfully.")
    print("Parameters:", model.count_params())

    return model


# --------------------------------------------------
# METRIC CALCULATION
# --------------------------------------------------

def calculate_metrics(reference, prediction):

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

    return psnr.numpy(), ssim.numpy(), mae.numpy()


# --------------------------------------------------
# MAIN EVALUATION
# --------------------------------------------------

def main():

    print("\n========================================")
    print(" NIGHTWATCH SIX-MODEL EVALUATION")
    print("========================================")

    print("\nLoading LOL test dataset...")

    _, _, test_ds = build_datasets()

    low_paths, _ = get_image_pairs("eval15")
    filenames = [Path(p).name for p in low_paths]

    print("Test images:", len(filenames))

    all_results = []
    summary_results = []
    errors = []

    for config in MODEL_CONFIGS:

        model_name = config["name"]

        print("\n----------------------------------------")
        print("Evaluating:", model_name)
        print("----------------------------------------")

        try:

            model = load_model(config)

            image_index = 0
            model_results = []

            for low_images, high_images in test_ds:

                predictions = model(
                    low_images,
                    training=False
                )

                predictions = tf.clip_by_value(
                    predictions, 0.0, 1.0
                )

                psnr, ssim, mae = calculate_metrics(
                    high_images,
                    predictions
                )

                for i in range(low_images.shape[0]):

                    if image_index >= len(filenames):
                        raise ValueError(
                            "More predictions than test filenames."
                        )

                    result = {
                        "Model": model_name,
                        "Image": filenames[image_index],
                        "PSNR": float(psnr[i]),
                        "SSIM": float(ssim[i]),
                        "MAE": float(mae[i])
                    }

                    model_results.append(result)
                    all_results.append(result)

                    image_index += 1

            if image_index != len(filenames):
                raise ValueError(
                    f"Expected {len(filenames)} images, "
                    f"evaluated {image_index}."
                )

            df = pd.DataFrame(model_results)

            model_csv = (
                RESULT_DIR / f"{model_name}_metrics.csv"
            )

            df.to_csv(model_csv, index=False)

            summary = {
                "Model": model_name,
                "Images_Evaluated": len(df),
                "Mean_PSNR": df["PSNR"].mean(),
                "Std_PSNR": df["PSNR"].std(),
                "Mean_SSIM": df["SSIM"].mean(),
                "Std_SSIM": df["SSIM"].std(),
                "Mean_MAE": df["MAE"].mean(),
                "Std_MAE": df["MAE"].std()
            }

            summary_results.append(summary)

            print("\nResults:")
            print(df[["PSNR", "SSIM", "MAE"]].mean())

            print("\nSaved:", model_csv)

            del model

        except Exception as e:

            print("\nERROR:", model_name)
            print(str(e))

            errors.append({
                "Model": model_name,
                "Error": str(e)
            })

    # --------------------------------------------------
    # SAVE COMBINED RESULTS
    # --------------------------------------------------

    if all_results:

        all_df = pd.DataFrame(all_results)

        all_df.to_csv(
            RESULT_DIR / "all_models_per_image.csv",
            index=False
        )

    if summary_results:

        summary_df = pd.DataFrame(summary_results)

        summary_df.to_csv(
            RESULT_DIR / "all_models_summary.csv",
            index=False
        )

        print("\n========================================")
        print(" FINAL MODEL COMPARISON")
        print("========================================")

        print(summary_df.round(5).to_string(index=False))

    if errors:

        pd.DataFrame(errors).to_csv(
            RESULT_DIR / "evaluation_errors.csv",
            index=False
        )

        print("\nSome models could not be evaluated.")
        print("See evaluation_errors.csv")

    print("\nEvaluation finished.")
    print("Results directory:", RESULT_DIR)


if __name__ == "__main__":
    main()
