
import cv2
import numpy as np


class AdaptiveRefinement:

    def __init__(
        self,
        target_brightness=0.36,
        min_gamma=0.75,
        max_gamma=1.0,
        clahe_clip=1.5,
        tile_grid=(8, 8),
        contrast_blend=0.35
    ):

        self.target_brightness = target_brightness
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma
        self.clahe_clip = clahe_clip
        self.tile_grid = tile_grid
        self.contrast_blend = contrast_blend

    def enhance(self, image):

        # Accept RGB image in range [0, 1]
        image = np.asarray(image, dtype=np.float32)

        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("Expected an RGB image with shape H x W x 3.")

        if not np.all(np.isfinite(image)):
            raise ValueError("Image contains invalid numerical values.")

        image = np.clip(image, 0.0, 1.0)

        # Convert RGB to YCrCb
        image_uint8 = np.round(image * 255).astype(np.uint8)

        ycrcb = cv2.cvtColor(
            image_uint8,
            cv2.COLOR_RGB2YCrCb
        )

        luminance = ycrcb[:, :, 0]

        # Estimate average luminance
        mean_brightness = float(np.mean(luminance)) / 255.0

        # Adaptive gamma correction
        if mean_brightness > 0.001:

            gamma = np.log(
                max(self.target_brightness, 0.001)
            ) / np.log(mean_brightness)

        else:
            gamma = self.max_gamma

        gamma = float(np.clip(
            gamma,
            self.min_gamma,
            self.max_gamma
        ))

        # Apply gamma correction only to luminance
        gamma_lut = np.array([
            ((i / 255.0) ** gamma) * 255
            for i in range(256)
        ]).clip(0, 255).astype(np.uint8)

        gamma_luminance = cv2.LUT(
            luminance,
            gamma_lut
        )

        # Local contrast enhancement
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip,
            tileGridSize=self.tile_grid
        )

        contrast_luminance = clahe.apply(
            gamma_luminance
        )

        # Blend contrast enhancement conservatively
        refined_luminance = cv2.addWeighted(
            gamma_luminance,
            1.0 - self.contrast_blend,
            contrast_luminance,
            self.contrast_blend,
            0
        )

        # Replace luminance while retaining chrominance
        ycrcb[:, :, 0] = refined_luminance

        refined_rgb = cv2.cvtColor(
            ycrcb,
            cv2.COLOR_YCrCb2RGB
        )

        refined_rgb = refined_rgb.astype(np.float32) / 255.0

        return np.clip(refined_rgb, 0.0, 1.0)


def adaptive_refine_batch(images, refiner=None):

    if refiner is None:
        refiner = AdaptiveRefinement()

    return np.stack([
        refiner.enhance(image)
        for image in images
    ], axis=0)
