"""Model architectures and losses.

Three complementary approaches, all mapping an RGB image (H, W, 3) in [0, 1]
to a per-pixel anomaly score (H, W, 1) in [0, 1]:

* ``baseline``    - small encoder-decoder CNN trained from scratch;
* ``transfer``    - U-Net decoder on an ImageNet-pretrained MobileNetV2 encoder;
* ``autoencoder`` - convolutional autoencoder trained to reconstruct *normal*
  road pixels only; the reconstruction error becomes the anomaly score
  (this follows the resynthesis idea of Lis et al., ICCV 2019).

Because the three share the same input/output contract, the evaluation code and
the Streamlit app treat them identically.
"""

from __future__ import annotations

import keras
import tensorflow as tf
from keras import layers

from . import config

ENCODER_LAYER_NAME = "mobilenetv2_encoder"
AUTOENCODER_LAYER_NAME = "conv_autoencoder"


# ---------------------------------------------------------------------------
# Losses (registered so that checkpoints reload without custom_objects)
# ---------------------------------------------------------------------------
@keras.saving.register_keras_serializable(package="roadanomaly")
def dice_loss(y_true, y_pred, smooth=1.0):
    y_true = tf.cast(y_true, y_pred.dtype)
    axes = (1, 2, 3)
    intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
    union = tf.reduce_sum(y_true, axis=axes) + tf.reduce_sum(y_pred, axis=axes)
    return 1.0 - (2.0 * intersection + smooth) / (union + smooth)


@keras.saving.register_keras_serializable(package="roadanomaly")
def bce_dice_loss(y_true, y_pred):
    """BCE optimises every pixel; Dice counters the 90/10 class imbalance."""
    bce = tf.reduce_mean(keras.losses.binary_crossentropy(y_true, y_pred), axis=(1, 2))
    return bce + dice_loss(y_true, y_pred)


@keras.saving.register_keras_serializable(package="roadanomaly")
def masked_reconstruction_loss(y_true, y_pred):
    """MSE computed on background pixels only.

    ``y_true`` = concat(image, anomaly_mask): the autoencoder never learns to
    reproduce anomalies, so they should be reconstructed poorly at test time.
    """
    image, anomaly = y_true[..., :3], y_true[..., 3:4]
    weights = 1.0 - anomaly
    squared = tf.reduce_mean(tf.square(image - y_pred), axis=-1, keepdims=True)
    return tf.reduce_sum(squared * weights, axis=(1, 2, 3)) / (tf.reduce_sum(weights, axis=(1, 2, 3)) + 1e-6)


LOSSES = {"bce": "binary_crossentropy", "bce_dice": bce_dice_loss, "dice": dice_loss}


# ---------------------------------------------------------------------------
# 1. Baseline CNN
# ---------------------------------------------------------------------------
def build_baseline_cnn(
    height=config.IMG_HEIGHT,
    width=config.IMG_WIDTH,
    base_filters=16,
    kernel_size=3,
    pool_size=2,
    batch_norm=False,
    dropout=0.0,
    l2=0.0,
):
    """Small encoder-decoder CNN.

    The three optional arguments implement the regularisation tools of the course (chapter 3):
    ``batch_norm`` (Conv -> BatchNorm -> ReLU), ``dropout`` (spatial dropout rate applied after every
    pooling stage and after the bottleneck) and ``l2`` (weight-decay coefficient lambda on the kernels).
    With the defaults the architecture is exactly the one of the reference experiment.
    """
    if height % pool_size**2 or width % pool_size**2:
        raise ValueError("Height and width must be divisible by 4.")
    regularizer = keras.regularizers.l2(l2) if l2 else None

    stack = [layers.Input((height, width, 3), name="rgb_image")]

    def conv(filters, name):
        stack.append(
            layers.Conv2D(
                filters, kernel_size, padding="same", activation=None if batch_norm else "relu",
                kernel_initializer="he_normal", kernel_regularizer=regularizer, name=name,
            )
        )
        if batch_norm:
            stack.append(layers.BatchNormalization(name=f"{name}_bn"))
            stack.append(layers.ReLU(name=f"{name}_relu"))

    def drop(name):
        if dropout:
            stack.append(layers.SpatialDropout2D(dropout, name=name))

    conv(base_filters, "encoder_conv_1")
    stack.append(layers.MaxPooling2D(pool_size, name="encoder_pool_1"))
    drop("dropout_1")
    conv(base_filters * 2, "encoder_conv_2")
    stack.append(layers.MaxPooling2D(pool_size, name="encoder_pool_2"))
    drop("dropout_2")
    conv(base_filters * 4, "bottleneck")
    drop("dropout_3")
    stack.append(layers.UpSampling2D(pool_size, interpolation="nearest", name="decoder_upsampling_1"))
    conv(base_filters * 2, "decoder_conv_1")
    stack.append(layers.UpSampling2D(pool_size, interpolation="nearest", name="decoder_upsampling_2"))
    conv(base_filters, "decoder_conv_2")
    stack.append(layers.Conv2D(1, 1, activation="sigmoid", name="anomaly_probability"))
    return keras.Sequential(stack, name="baseline_segmentation_cnn")


# ---------------------------------------------------------------------------
# 2. Transfer learning: MobileNetV2 U-Net
# ---------------------------------------------------------------------------
FEATURE_LAYERS = (
    "block_1_expand_relu",  # 1/2
    "block_3_expand_relu",  # 1/4
    "block_6_expand_relu",  # 1/8
    "block_13_expand_relu",  # 1/16
    "block_16_project",  # 1/32
)


def _decoder_block(x, skip, filters, name):
    x = layers.UpSampling2D(2, interpolation="nearest", name=f"{name}_upsampling")(x)
    x = layers.Concatenate(name=f"{name}_concatenation")([x, skip])
    for i in (1, 2):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu", kernel_initializer="he_normal", name=f"{name}_conv_{i}")(x)
    return x


def build_transfer_unet(
    height=config.IMG_HEIGHT,
    width=config.IMG_WIDTH,
    decoder_filters=(128, 64, 32, 16),
    final_filters=8,
    weights="imagenet",
    freeze_encoder=True,
):
    stride = 32
    pad_h, pad_w = (-height) % stride, (-width) % stride
    backbone = keras.applications.MobileNetV2(
        input_shape=(height + pad_h, width + pad_w, 3), include_top=False, weights=weights, alpha=1.0
    )
    encoder = keras.Model(
        backbone.input, [backbone.get_layer(n).output for n in FEATURE_LAYERS], name=ENCODER_LAYER_NAME
    )
    encoder.trainable = not freeze_encoder

    inputs = layers.Input((height, width, 3), name="transfer_rgb_image")
    x = inputs
    if pad_h or pad_w:
        x = layers.ZeroPadding2D(((0, pad_h), (0, pad_w)), name="input_padding")(x)
    # Pipeline images are in [0, 1]; MobileNetV2 expects [-1, 1].
    x = layers.Rescaling(2.0, offset=-1.0, name="mobilenetv2_rescaling")(x)
    features = encoder(x, training=False)  # BatchNorm stays in inference mode, even when fine-tuning
    x, skips = features[-1], list(reversed(features[:-1]))
    for i, (filters, skip) in enumerate(zip(decoder_filters, skips), start=1):
        x = _decoder_block(x, skip, filters, f"decoder_block_{i}")
    x = layers.UpSampling2D(2, interpolation="nearest", name="final_upsampling")(x)
    x = layers.Conv2D(final_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal", name="final_features")(x)
    x = layers.Conv2D(1, 1, activation="sigmoid", name="padded_anomaly_probability")(x)
    if pad_h or pad_w:
        x = layers.Cropping2D(((0, pad_h), (0, pad_w)), name="remove_padding")(x)
    return keras.Model(inputs, x, name="mobilenetv2_unet")


def unfreeze_encoder_top(model, n_layers: int = 30) -> int:
    """Unfreeze the last ``n_layers`` of the encoder (BatchNorm layers stay frozen)."""
    encoder = model.get_layer(ENCODER_LAYER_NAME)
    encoder.trainable = True
    for layer in encoder.layers[:-n_layers]:
        layer.trainable = False
    for layer in encoder.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    return sum(int(tf.size(w)) for w in encoder.trainable_weights)


# ---------------------------------------------------------------------------
# 3. Convolutional autoencoder (complementary, unsupervised-style approach)
# ---------------------------------------------------------------------------
def build_conv_autoencoder(height=config.IMG_HEIGHT, width=config.IMG_WIDTH, filters=(32, 64, 128), latent_channels=8):
    factor = 2 ** len(filters)
    if height % factor or width % factor:
        raise ValueError(f"Height and width must be divisible by {factor}.")
    inputs = layers.Input((height, width, 3), name="ae_rgb_image")
    x = inputs
    for i, f in enumerate(filters, start=1):
        x = layers.Conv2D(f, 3, padding="same", activation="relu", name=f"ae_enc_conv_{i}a")(x)
        x = layers.Conv2D(f, 3, strides=2, padding="same", activation="relu", name=f"ae_enc_conv_{i}b")(x)
    # Narrow bottleneck: forces the network to learn typical road appearance
    # instead of copying the input (no skip connections on purpose).
    x = layers.Conv2D(latent_channels, 1, activation="relu", name="ae_latent")(x)
    for i, f in enumerate(reversed(filters), start=1):
        x = layers.Conv2DTranspose(f, 3, strides=2, padding="same", activation="relu", name=f"ae_dec_deconv_{i}")(x)
        x = layers.Conv2D(f, 3, padding="same", activation="relu", name=f"ae_dec_conv_{i}")(x)
    outputs = layers.Conv2D(3, 1, activation="sigmoid", name="ae_reconstruction")(x)
    return keras.Model(inputs, outputs, name=AUTOENCODER_LAYER_NAME)


def build_autoencoder_scorer(autoencoder, scale: float, smooth: int = 9, height=config.IMG_HEIGHT, width=config.IMG_WIDTH):
    """Wrap the autoencoder into a model returning an anomaly score in [0, 1].

    score = clip( avg_pool( mean_c (x - x_hat)^2 ) / scale , 0, 1 )
    ``scale`` is a high percentile of the background error on the training set,
    so a score of 1 means "as badly reconstructed as the worst 1% of road pixels".
    Only standard Keras layers are used, so the model saves/loads as a plain .keras file.
    """
    inputs = layers.Input((height, width, 3), name="scorer_rgb_image")
    reconstruction = autoencoder(inputs)
    diff = layers.Subtract(name="reconstruction_residual")([inputs, reconstruction])
    squared = layers.Multiply(name="squared_residual")([diff, diff])
    error = layers.Conv2D(
        1, 1, use_bias=False, trainable=False, kernel_initializer=keras.initializers.Constant(1.0 / 3.0), name="channel_mean"
    )(squared)
    error = layers.AveragePooling2D(smooth, strides=1, padding="same", name="spatial_smoothing")(error)
    score = layers.Rescaling(1.0 / float(scale), name="normalise_by_background_error")(error)
    score = layers.ReLU(max_value=1.0, name="anomaly_score")(score)
    return keras.Model(inputs, score, name="autoencoder_anomaly_scorer")


def count_parameters(model) -> dict:
    trainable = sum(int(tf.size(w)) for w in model.trainable_weights)
    frozen = sum(int(tf.size(w)) for w in model.non_trainable_weights)
    return {"trainable": trainable, "non_trainable": frozen, "total": trainable + frozen}
