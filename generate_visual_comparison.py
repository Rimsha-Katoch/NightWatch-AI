
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from models.adaptive_refinement import AdaptiveRefinement
from data_loader import build_datasets, get_image_pairs


# ==================================================
# CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR / "checkpoints" / "best_resunet_exp5.keras"
)

OUTPUT_DIR = (
    BASE_DIR / "results" / "visual_comparison_final"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
# CREATE COMPARISON FIGURE
# ==================================================

def save_individual_comparison(
    filename,
    images,
    output_path
):

    titles = [
        "Low-light Input",
        "Experiment 5",
        "Adaptive Gamma + CLAHE",
        "Full NightWatch",
        "Ground Truth"
    ]

    fig, axes = plt.subplots(
        1, 5,
        figsize=(20, 5)
    )

    for ax, title, image in zip(
        axes, titles, images
    ):

        ax.imshow(
            np.clip(image, 0, 1)
        )

        ax.set_title(
            title,
            fontsize=10
        )

        ax.axis("off")

    fig.suptitle(
        f"NightWatch Visual Comparison: {filename}",
        fontsize=14
    )

    plt.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)


# ==================================================
# CREATE CONTACT SHEET
# ==================================================

def save_contact_sheet(page_number, page_data):

    titles = [
        "Low-light Input",
        "Experiment 5",
        "Adaptive Gamma + CLAHE",
        "Full NightWatch",
        "Ground Truth"
    ]

    rows = len(page_data)

    fig, axes = plt.subplots(
        rows,
        5,
        figsize=(20, 4 * rows),
        squeeze=False
    )

    for row_index, item in enumerate(page_data):

        for col_index, image in enumerate(item["images"]):

            ax = axes[row_index, col_index]

            ax.imshow(
                np.clip(image, 0, 1)
            )

            ax.axis("off")

            if row_index == 0:
                ax.set_title(
                    titles[col_index],
                    fontsize=10
                )

        axes[row_index, 0].set_ylabel(
            item["filename"],
            fontsize=8,
            rotation=0,
            labelpad=55,
            va="center"
        )

    fig.suptitle(
        f"NightWatch Visual Comparison - Page {page_number}",
        fontsize=15
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR /
        f"contact_sheet_page_{page_number}.png"
    )

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)


# ==================================================
# MAIN
# ==================================================

def main():

    print("\n========================================")
    print(" NIGHTWATCH VISUAL COMPARISON")
    print("========================================")

    print("\nLoading test dataset...")

    _, _, test_ds = build_datasets()

    low_paths, _ = get_image_pairs("eval15")
    filenames = [Path(p).name for p in low_paths]

    print("Test images:", len(filenames))

    print("\nLoading Experiment 5...")

    model = tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False
    )

    print("Model loaded successfully.")

    refiner = AdaptiveRefinement()

    page_data = []
    image_index = 0

    for low_batch, high_batch in test_ds:

        predictions = model(
            low_batch,
            training=False
        )

        predictions = tf.clip_by_value(
            predictions, 0, 1
        ).numpy()

        low_np = low_batch.numpy()
        high_np = high_batch.numpy()

        for i in range(len(predictions)):

            if image_index >= len(filenames):
                raise ValueError(
                    "More predictions than test filenames."
                )

            low_image = low_np[i]
            model_output = predictions[i]
            ground_truth = high_np[i]

            # Adaptive gamma correction + CLAHE
            adaptive_output = refiner.enhance(
                model_output
            )

            # Complete NightWatch pipeline
            final_output = adjust_saturation(
                adaptive_output,
                SATURATION_SCALE
            )

            filename = filenames[image_index]

            comparison_images = [
                low_image,
                model_output,
                adaptive_output,
                final_output,
                ground_truth
            ]

            # Save individual comparison
            individual_path = (
                OUTPUT_DIR /
                f"comparison_{image_index + 1:02d}.png"
            )

            save_individual_comparison(
                filename,
                comparison_images,
                individual_path
            )

            page_data.append({
                "filename": filename,
                "images": comparison_images
            })

            image_index += 1

            print(
                f"Generated comparison "
                f"{image_index}/{len(filenames)}: {filename}"
            )

            # Save a contact sheet after every 5 images
            if len(page_data) == 5:

                page_number = (
                    (image_index - 1) // 5 + 1
                )

                save_contact_sheet(
                    page_number,
                    page_data
                )

                page_data = []

    if image_index != len(filenames):
        raise ValueError(
            f"Expected {len(filenames)} images, "
            f"processed {image_index}."
        )

    # Save any remaining images
    if page_data:

        page_number = (
            (image_index - 1) // 5 + 1
        )

        save_contact_sheet(
            page_number,
            page_data
        )

    print("\n========================================")
    print(" VISUAL COMPARISON COMPLETED")
    print("========================================")

    print("Images processed:", image_index)
    print("Output folder:", OUTPUT_DIR)

    print("\nGenerated:")
    print("- 15 individual comparison images")
    print("- 3 contact sheets")


if __name__ == "__main__":
    main()
