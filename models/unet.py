import tensorflow as tf
from tensorflow.keras import layers, Model


def conv_block(inputs, filters):

    x = layers.Conv2D(
        filters, 3, padding="same",
        kernel_initializer="he_normal"
    )(inputs)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(
        filters, 3, padding="same",
        kernel_initializer="he_normal"
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    return x


def build_unet(input_shape=(256, 384, 3)):

    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = conv_block(inputs, 32)
    p1 = layers.MaxPooling2D()(c1)

    c2 = conv_block(p1, 64)
    p2 = layers.MaxPooling2D()(c2)

    c3 = conv_block(p2, 128)
    p3 = layers.MaxPooling2D()(c3)

    c4 = conv_block(p3, 256)
    p4 = layers.MaxPooling2D()(c4)

    # Bottleneck
    b = conv_block(p4, 512)

    # Decoder
    u4 = layers.Conv2DTranspose(
        256, 2, strides=2, padding="same"
    )(b)

    u4 = layers.Concatenate()([u4, c4])
    c5 = conv_block(u4, 256)

    u3 = layers.Conv2DTranspose(
        128, 2, strides=2, padding="same"
    )(c5)

    u3 = layers.Concatenate()([u3, c3])
    c6 = conv_block(u3, 128)

    u2 = layers.Conv2DTranspose(
        64, 2, strides=2, padding="same"
    )(c6)

    u2 = layers.Concatenate()([u2, c2])
    c7 = conv_block(u2, 64)

    u1 = layers.Conv2DTranspose(
        32, 2, strides=2, padding="same"
    )(c7)

    u1 = layers.Concatenate()([u1, c1])
    c8 = conv_block(u1, 32)

    # Enhanced RGB image
    outputs = layers.Conv2D(
        3, 1, activation="sigmoid"
    )(c8)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="NightWatch_UNet"
    )

    return model
