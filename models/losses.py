import tensorflow as tf


def combined_loss(y_true, y_pred):

    # Pixel-level L1 loss
    l1_loss = tf.reduce_mean(
        tf.abs(y_true - y_pred)
    )

    # Structural Similarity Index
    ssim_value = tf.image.ssim(
        y_true,
        y_pred,
        max_val=1.0
    )

    ssim_loss = 1.0 - tf.reduce_mean(ssim_value)

    # Weighted combined loss
    total_loss = (
        0.8 * l1_loss +
        0.2 * ssim_loss
    )

    return total_loss


def calculate_metrics(y_true, y_pred):

    psnr = tf.reduce_mean(
        tf.image.psnr(y_true, y_pred, max_val=1.0)
    )

    ssim = tf.reduce_mean(
        tf.image.ssim(y_true, y_pred, max_val=1.0)
    )

    return psnr, ssim
