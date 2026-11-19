import tensorflow as tf
from tensorflow.keras import layers, Model


# ==========================================
# 1. RESIDUAL BLOCK
# ==========================================

def residual_block(inputs, filters):

    shortcut = inputs

    # First convolution
    x = layers.Conv2D(
        filters,
        kernel_size=3,
        padding="same",
        kernel_initializer="he_normal"
    )(inputs)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Second convolution
    x = layers.Conv2D(
        filters,
        kernel_size=3,
        padding="same",
        kernel_initializer="he_normal"
    )(x)

    x = layers.BatchNormalization()(x)

    # Adjust shortcut channels if required
    if inputs.shape[-1] != filters:

        shortcut = layers.Conv2D(
            filters,
            kernel_size=1,
            padding="same"
        )(shortcut)

        shortcut = layers.BatchNormalization()(shortcut)

    # Residual connection
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)

    return x


# ==========================================
# 2. ATTENTION GATE
# ==========================================

def attention_gate(skip, gate, filters):

    # Transform encoder features
    theta = layers.Conv2D(
        filters,
        kernel_size=1,
        padding="same"
    )(skip)

    # Transform decoder features
    phi = layers.Conv2D(
        filters,
        kernel_size=1,
        padding="same"
    )(gate)

    # Combine features
    combined = layers.Add()([theta, phi])
    combined = layers.ReLU()(combined)

    # Generate attention coefficients
    attention = layers.Conv2D(
        1,
        kernel_size=1,
        activation="sigmoid"
    )(combined)

    # Apply attention to skip features
    attended = layers.Multiply()([
        skip,
        attention
    ])

    return attended


# ==========================================
# 3. DECODER BLOCK
# ==========================================

def decoder_block(inputs, skip, filters):

    # Upsampling
    x = layers.Conv2DTranspose(
        filters,
        kernel_size=2,
        strides=2,
        padding="same"
    )(inputs)

    # Attention-filtered skip connection
    attended_skip = attention_gate(
        skip,
        x,
        filters
    )

    # Concatenate features
    x = layers.Concatenate()([
        x,
        attended_skip
    ])

    # Residual feature refinement
    x = residual_block(x, filters)

    return x


# ==========================================
# 4. RESIDUAL ATTENTION U-NET
# ==========================================

def build_resunet(input_shape=(256, 384, 3)):

    inputs = layers.Input(
        shape=input_shape,
        name="low_light_input"
    )

    # --------------------------------------
    # ENCODER
    # --------------------------------------

    c1 = residual_block(inputs, 16)
    p1 = layers.MaxPooling2D()(c1)

    c2 = residual_block(p1, 32)
    p2 = layers.MaxPooling2D()(c2)

    c3 = residual_block(p2, 64)
    p3 = layers.MaxPooling2D()(c3)

    c4 = residual_block(p3, 128)
    p4 = layers.MaxPooling2D()(c4)

    # --------------------------------------
    # BOTTLENECK
    # --------------------------------------

    bottleneck = residual_block(p4, 256)

    # --------------------------------------
    # DECODER
    # --------------------------------------

    d4 = decoder_block(
        bottleneck, c4, 128
    )

    d3 = decoder_block(
        d4, c3, 64
    )

    d2 = decoder_block(
        d3, c2, 32
    )

    d1 = decoder_block(
        d2, c1, 16
    )

    # --------------------------------------
    # RESIDUAL IMAGE CORRECTION
    # --------------------------------------

    residual = layers.Conv2D(
        filters=3,
        kernel_size=3,
        padding="same",
        activation="tanh",
        name="residual_prediction"
    )(d1)

    # Limit correction magnitude
    residual = layers.Rescaling(
        scale=0.25,
        name="correction_scaling"
    )(residual)

    # Add predicted correction to original input
    corrected = layers.Add(
        name="image_correction"
    )([inputs, residual])

    # Restrict final pixel values to [0, 1]
    outputs = layers.ReLU(
        max_value=1.0,
        name="residual_image_output"
    )(corrected)

    # --------------------------------------
    # BUILD MODEL
    # --------------------------------------

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="NightWatch_Residual_Attention_UNet"
    )

    return model
