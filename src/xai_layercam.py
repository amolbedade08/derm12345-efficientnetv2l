import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "EfficientNetV2L",
    "models",
    "final_model.keras"
)

XAI_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "EfficientNetV2L",
    "xai"
)

os.makedirs(XAI_DIR, exist_ok=True)

IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "acb", "acd", "ajb", "ajd", "ak",
    "alm", "angk", "anm", "bcc", "bd",
    "bdb", "cb", "ccb", "ccd", "cd",
    "ch", "cjb", "db", "df", "dfsp",
    "ha", "isl", "jb", "jd", "ks",
    "la", "lk", "lm", "lmm", "ls",
    "mcb", "mel", "mpd", "pg", "rd",
    "sa", "scc", "sk", "sl", "srjd"
]


# ============================================================
# LOAD MODEL
# ============================================================

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded.")


# ============================================================
# FIND CONVOLUTIONAL LAYERS
# ============================================================

conv_layers = []

for layer in model.layers:

    if isinstance(
        layer,
        tf.keras.layers.Conv2D
    ):

        conv_layers.append(layer)


print(
    "Number of Conv2D layers:",
    len(conv_layers)
)

if len(conv_layers) == 0:

    raise RuntimeError(
        "No Conv2D layers found."
    )


# Use several late layers for multi-layer explanation
selected_layers = conv_layers[-3:]

print("\nSelected layers:")

for layer in selected_layers:
    print(layer.name)


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(path):

    image = tf.io.read_file(path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image = tf.image.resize(
        image,
        IMG_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    ) / 255.0

    return image


# ============================================================
# LAYERCAM
# ============================================================

def layercam(
    model,
    image,
    target_class,
    layer
):

    image = tf.expand_dims(
        image,
        axis=0
    )

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            layer.output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_output, predictions = (
            grad_model(image)
        )

        target_score = predictions[
            :, target_class
        ]

    gradients = tape.gradient(
        target_score,
        conv_output
    )

    # Positive gradients
    positive_gradients = tf.nn.relu(
        gradients
    )

    # LayerCAM weighting
    activation = (
        conv_output *
        positive_gradients
    )

    heatmap = tf.reduce_sum(
        activation,
        axis=-1
    )

    heatmap = tf.nn.relu(
        heatmap
    )

    heatmap = heatmap[0]

    # Normalize
    heatmap = (
        heatmap /
        (
            tf.reduce_max(
                heatmap
            ) + 1e-8
        )
    )

    heatmap = tf.image.resize(
        heatmap[..., tf.newaxis],
        IMG_SIZE
    )

    return heatmap.numpy().squeeze()


# ============================================================
# MULTI-LAYER LAYERCAM
# ============================================================

def multi_layer_layercam(
    model,
    image
):

    image_batch = tf.expand_dims(
        image,
        axis=0
    )

    predictions = model.predict(
        image_batch,
        verbose=0
    )

    predicted_class = int(
        np.argmax(
            predictions[0]
        )
    )

    heatmaps = []

    for layer in selected_layers:

        heatmap = layercam(
            model,
            image,
            predicted_class,
            layer
        )

        heatmaps.append(
            heatmap
        )

    # Average multi-layer explanations
    final_heatmap = np.mean(
        heatmaps,
        axis=0
    )

    final_heatmap = (
        final_heatmap /
        (
            np.max(
                final_heatmap
            ) + 1e-8
        )
    )

    return (
        predicted_class,
        predictions[0],
        final_heatmap
    )


# ============================================================
# VISUALIZATION
# ============================================================

def save_explanation(
    image_path
):

    image = load_image(
        image_path
    )

    predicted_class, probabilities, heatmap = (
        multi_layer_layercam(
            model,
            image
        )
    )

    original = image.numpy()

    plt.figure(
        figsize=(12, 4)
    )

    plt.subplot(
        1, 3, 1
    )

    plt.imshow(
        original
    )

    plt.title("Original")
    plt.axis("off")

    plt.subplot(
        1, 3, 2
    )

    plt.imshow(
        heatmap,
        cmap="jet"
    )

    plt.title("LayerCAM")
    plt.axis("off")

    plt.subplot(
        1, 3, 3
    )

    plt.imshow(
        original
    )

    plt.imshow(
        heatmap,
        cmap="jet",
        alpha=0.45
    )

    plt.title(
        f"Prediction: "
        f"{CLASS_NAMES[predicted_class]}\n"
        f"Confidence: "
        f"{probabilities[predicted_class]:.3f}"
    )

    plt.axis("off")

    filename = os.path.splitext(
        os.path.basename(
            image_path
        )
    )[0]

    output_path = os.path.join(
        XAI_DIR,
        filename + "_LayerCAM.png"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved:",
        output_path
    )


# ============================================================
# CHANGE THIS TO AN IMAGE
# ============================================================

IMAGE_PATH = r"PUT_IMAGE_PATH_HERE"

if not os.path.exists(
    IMAGE_PATH
):

    raise FileNotFoundError(
        "Set IMAGE_PATH to an actual "
        "DERM12345 image."
    )

save_explanation(
    IMAGE_PATH
)

print("\nLayerCAM complete.")