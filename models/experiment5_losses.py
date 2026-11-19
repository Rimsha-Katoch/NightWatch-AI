
import tensorflow as tf

def build_experiment5_loss():

    def calculate_components(y_true, y_pred):

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)

        # L1 reconstruction loss
        l1_loss = tf.reduce_mean(
            tf.abs(y_true - y_pred)
        )

        # Structural similarity loss
        ssim_loss = 1.0 - tf.reduce_mean(
            tf.image.ssim(y_true, y_pred, max_val=1.0)
        )

        # Brightness consistency
        true_gray = tf.image.rgb_to_grayscale(y_true)
        pred_gray = tf.image.rgb_to_grayscale(y_pred)

        true_brightness = tf.reduce_mean(
            true_gray, axis=[1, 2, 3]
        )
        pred_brightness = tf.reduce_mean(
            pred_gray, axis=[1, 2, 3]
        )

        brightness_loss = tf.reduce_mean(
            tf.abs(true_brightness - pred_brightness)
        )

        # Contrast consistency
        true_contrast = tf.math.reduce_std(
            true_gray, axis=[1, 2, 3]
        )
        pred_contrast = tf.math.reduce_std(
            pred_gray, axis=[1, 2, 3]
        )

        contrast_loss = tf.reduce_mean(
            tf.abs(true_contrast - pred_contrast)
        )

        # Saturation consistency
        true_hsv = tf.image.rgb_to_hsv(y_true)
        pred_hsv = tf.image.rgb_to_hsv(y_pred)

        saturation_loss = tf.reduce_mean(
            tf.abs(true_hsv[..., 1] - pred_hsv[..., 1])
        )

        return {
            "l1": l1_loss,
            "ssim": ssim_loss,
            "brightness": brightness_loss,
            "contrast": contrast_loss,
            "saturation": saturation_loss
        }

    def experiment5_loss(y_true, y_pred):

        c = calculate_components(y_true, y_pred)

        return (
            0.55 * c["l1"]
            + 0.25 * c["ssim"]
            + 0.08 * c["brightness"]
            + 0.06 * c["contrast"]
            + 0.06 * c["saturation"]
        )

    return experiment5_loss, calculate_components
