import tensorflow as tf


def build_advanced_loss(input_shape=(256, 384, 3)):

    # Load pretrained VGG16 for perceptual feature extraction
    vgg = tf.keras.applications.VGG16(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape
    )

    # Use intermediate features
    feature_extractor = tf.keras.Model(
        inputs=vgg.input,
        outputs=vgg.get_layer("block3_conv3").output
    )

    feature_extractor.trainable = False

    def calculate_components(y_true, y_pred):

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # 1. Pixel reconstruction loss
        l1_loss = tf.reduce_mean(
            tf.abs(y_true - y_pred)
        )

        # 2. Structural similarity loss
        ssim_loss = 1.0 - tf.reduce_mean(
            tf.image.ssim(
                y_true,
                y_pred,
                max_val=1.0
            )
        )

        # 3. Perceptual loss
        true_vgg = tf.keras.applications.vgg16.preprocess_input(
            y_true * 255.0
        )

        pred_vgg = tf.keras.applications.vgg16.preprocess_input(
            y_pred * 255.0
        )

        true_features = tf.stop_gradient(
            feature_extractor(true_vgg, training=False)
        )

        pred_features = feature_extractor(
            pred_vgg,
            training=False
        )

        perceptual_difference = tf.reduce_mean(
            tf.abs(true_features - pred_features)
        )

        # Normalize perceptual difference
        feature_scale = tf.reduce_mean(
            tf.abs(true_features)
        ) + 1e-6

        perceptual_loss = (
            perceptual_difference / feature_scale
        )

        # 4. Colour consistency loss
        true_mean, true_variance = tf.nn.moments(
            y_true,
            axes=[1, 2]
        )

        pred_mean, pred_variance = tf.nn.moments(
            y_pred,
            axes=[1, 2]
        )

        true_std = tf.sqrt(true_variance + 1e-6)
        pred_std = tf.sqrt(pred_variance + 1e-6)

        mean_loss = tf.reduce_mean(
            tf.abs(true_mean - pred_mean)
        )

        std_loss = tf.reduce_mean(
            tf.abs(true_std - pred_std)
        )

        color_loss = mean_loss + std_loss

        return {
            "l1": l1_loss,
            "ssim": ssim_loss,
            "perceptual": perceptual_loss,
            "color": color_loss
        }

    def advanced_loss(y_true, y_pred):

        components = calculate_components(
            y_true, y_pred
        )

        total_loss = (
            0.70 * components["l1"]
            + 0.20 * components["ssim"]
            + 0.05 * components["perceptual"]
            + 0.05 * components["color"]
        )

        return total_loss

    return advanced_loss, calculate_components
