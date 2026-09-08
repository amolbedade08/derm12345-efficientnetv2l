import os
import gc

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "EfficientNetV2L",
    "models",
    "efficientnetv2l_final.keras"
)

PREDICTIONS_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "EfficientNetV2L",
    "external_PAD_UFES20",
    "predictions.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "EfficientNetV2L",
    "XAI"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_SIZE = (224, 224)

N_CORRECT_PER_CLASS = 1
N_INCORRECT_PER_CLASS = 1

TARGET_CLASSES = [
    "BCC",
    "MEL",
    "SCC",
    "SEK"
]


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("EfficientNetV2-L CORRECT NESTED-MODEL LayerCAM")
print("=" * 70)

print("TensorFlow:", tf.__version__)

# Force CPU.
print("XAI device: CPU")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model input :", model.input_shape)
print("Model output:", model.output_shape)
print("Parameters  :", model.count_params())


# ============================================================
# PRINT TOP LEVEL ARCHITECTURE
# ============================================================

print("\nTop-level model layers:")

for i, layer in enumerate(model.layers):
    print(
        i,
        layer.name,
        type(layer).__name__
    )


# ============================================================
# FIND BACKBONE
# ============================================================

backbone = None
backbone_index = None

for i, layer in enumerate(model.layers):

    if isinstance(
        layer,
        tf.keras.Model
    ):
        backbone = layer
        backbone_index = i
        break

if backbone is None:
    raise RuntimeError(
        "Nested EfficientNetV2-L backbone not found."
    )

print("\nBackbone:")
print(backbone.name)

print(
    "Backbone input:",
    backbone.input_shape
)

print(
    "Backbone output:",
    backbone.output_shape
)


# ============================================================
# IDENTIFY CLASSIFICATION HEAD
# ============================================================

head_layers = model.layers[
    backbone_index + 1:
]

print("\nClassification head:")

for layer in head_layers:
    print(
        " -",
        layer.name,
        type(layer).__name__
    )


# ============================================================
# GET CONV LAYERS
# ============================================================

def collect_conv_layers(root):

    layers_found = []

    for layer in root.layers:

        if isinstance(
            layer,
            (
                tf.keras.layers.Conv2D,
                tf.keras.layers.DepthwiseConv2D,
                tf.keras.layers.SeparableConv2D
            )
        ):
            layers_found.append(layer)

        if isinstance(
            layer,
            tf.keras.Model
        ):
            layers_found.extend(
                collect_conv_layers(layer)
            )

    return layers_found


conv_layers = collect_conv_layers(backbone)

print(
    "\nTotal convolutional layers:",
    len(conv_layers)
)


# ============================================================
# SELECT REAL SPATIAL CONVOLUTION LAYERS
# ============================================================

project_layers = [
    layer
    for layer in conv_layers
    if layer.name.endswith("project_conv")
]

top_conv_layers = [
    layer
    for layer in conv_layers
    if layer.name == "top_conv"
]

selected_layers = []

if project_layers:
    selected_layers.append(
        project_layers[-1]
    )

if top_conv_layers:
    selected_layers.append(
        top_conv_layers[-1]
    )

# Remove duplicates
unique_layers = []
seen = set()

for layer in selected_layers:

    if layer.name not in seen:

        unique_layers.append(layer)
        seen.add(layer.name)

selected_layers = unique_layers

if not selected_layers:

    raise RuntimeError(
        "No suitable LayerCAM layers found."
    )

print("\nSelected LayerCAM layers:")

for layer in selected_layers:
    print(
        " -",
        layer.name
    )

pd.DataFrame(
    {
        "selected_layer": [
            layer.name
            for layer in selected_layers
        ]
    }
).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "selected_layercam_layers.csv"
    ),
    index=False
)


# ============================================================
# CLASS MAPPING
# ============================================================

MAPPING_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "EfficientNetV2L",
    "csv_results",
    "final_class_mapping.csv"
)

mapping_df = pd.read_csv(
    MAPPING_PATH
).sort_values(
    "class_index"
)

model_class_names = (
    mapping_df["label"]
    .astype(str)
    .tolist()
)

class_to_index = {
    str(row["label"]): int(
        row["class_index"]
    )
    for _, row in mapping_df.iterrows()
}

print("\nShared class indices:")

for cls in [
    "bcc",
    "mel",
    "scc",
    "sk"
]:

    print(
        f"{cls}: {class_to_index[cls]}"
    )


# ============================================================
# LOAD EXTERNAL PREDICTIONS
# ============================================================

pred_df = pd.read_csv(
    PREDICTIONS_PATH
)

required_columns = [
    "image_path",
    "diagnostic",
    "mapped_label",
    "true_class_index",
    "predicted_class_index",
    "predicted_class"
]

for column in required_columns:

    if column not in pred_df.columns:

        raise RuntimeError(
            f"Missing required column: {column}"
        )

pred_df["correct"] = (
    pred_df["true_class_index"]
    ==
    pred_df["predicted_class_index"]
)

print(
    "\nPrediction rows:",
    len(pred_df)
)

print(
    "Correct:",
    int(pred_df["correct"].sum())
)

print(
    "Incorrect:",
    int((~pred_df["correct"]).sum())
)


# ============================================================
# IMAGE LOADER
# ============================================================

def load_image(path):

    raw = tf.io.read_file(path)

    image = tf.image.decode_png(
        raw,
        channels=3
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    # Match training:
    # model contains EfficientNetV2 preprocessing.
    image = tf.cast(
        image,
        tf.float32
    )

    return image


# ============================================================
# APPLY ORIGINAL CLASSIFICATION HEAD
# ============================================================

def apply_classification_head(
    backbone_output,
    training=False
):
    """
    Apply exactly the layers after the nested backbone
    from the saved model.
    """

    x = backbone_output

    for layer in head_layers:

        x = layer(
            x,
            training=training
        )

    return x


# ============================================================
# SINGLE LAYER LAYERCAM
# ============================================================

def calculate_single_layer_cam(
    image_tensor,
    target_class,
    target_layer
):

    print(
        "    Computing:",
        target_layer.name
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Build a model INSIDE the nested EfficientNet backbone:
    #
    # backbone.input
    #       |
    #       +--> target internal layer
    #       |
    #       +--> backbone.output
    #
    # Then apply the original classifier head manually.
    # This preserves the actual gradient path.
    # --------------------------------------------------------

    intermediate_model = tf.keras.Model(
        inputs=backbone.input,
        outputs=[
            target_layer.output,
            backbone.output
        ]
    )

    image_batch = tf.expand_dims(
        image_tensor,
        axis=0
    )

    with tf.GradientTape() as tape:

        feature_map, backbone_output = (
            intermediate_model(
                image_batch,
                training=False
            )
        )

        predictions = (
            apply_classification_head(
                backbone_output,
                training=False
            )
        )

        target_score = predictions[
            0,
            target_class
        ]

    gradients = tape.gradient(
        target_score,
        feature_map
    )

    if gradients is None:

        raise RuntimeError(
            "Gradient is None for layer "
            f"{target_layer.name}. "
            "The selected layer is not connected "
            "to the classifier."
        )

    # --------------------------------------------------------
    # LayerCAM
    # --------------------------------------------------------

    positive_gradients = tf.maximum(
        gradients,
        0.0
    )

    weighted_activations = (
        positive_gradients
        *
        feature_map
    )

    cam = tf.reduce_sum(
        weighted_activations,
        axis=-1
    )

    cam = tf.maximum(
        cam,
        0.0
    )

    cam = tf.expand_dims(
        cam,
        axis=-1
    )

    cam = tf.image.resize(
        cam,
        IMAGE_SIZE,
        method="bilinear"
    )

    cam = tf.squeeze(
        cam,
        axis=-1
    )

    cam = cam.numpy()[0]

    # Normalize
    cam_min = np.min(cam)
    cam_max = np.max(cam)

    if cam_max > cam_min:

        cam = (
            cam - cam_min
        ) / (
            cam_max - cam_min
        )

    else:

        cam = np.zeros_like(cam)

    probabilities = (
        predictions
        .numpy()[0]
    )

    # Release intermediate graph.
    del intermediate_model
    del tape
    del feature_map
    del backbone_output
    del predictions
    del gradients
    del weighted_activations

    gc.collect()

    return cam, probabilities


# ============================================================
# MULTI-LAYER LAYERCAM
# ============================================================

def generate_multilayer_layercam(
    image_tensor,
    target_class
):

    cams = []
    probabilities = None

    for layer in selected_layers:

        cam, probabilities = (
            calculate_single_layer_cam(
                image_tensor,
                target_class,
                layer
            )
        )

        cams.append(cam)

        gc.collect()

    fused_cam = np.mean(
        np.stack(cams),
        axis=0
    )

    fused_min = fused_cam.min()
    fused_max = fused_cam.max()

    if fused_max > fused_min:

        fused_cam = (
            fused_cam - fused_min
        ) / (
            fused_max - fused_min
        )

    else:

        fused_cam = np.zeros_like(
            fused_cam
        )

    return (
        fused_cam,
        probabilities
    )


# ============================================================
# SAVE FIGURE
# ============================================================

def save_xai_figure(
    image_tensor,
    cam,
    true_label,
    predicted_label,
    confidence,
    output_path,
    status
):

    original = (
        image_tensor.numpy()
        / 255.0
    )

    original = np.clip(
        original,
        0,
        1
    )

    heatmap = plt.get_cmap(
        "jet"
    )(cam)[..., :3]

    overlay = (
        0.55 * original
        +
        0.45 * heatmap
    )

    overlay = np.clip(
        overlay,
        0,
        1
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    axes[0].imshow(
        original
    )

    axes[0].set_title(
        f"Original\nTrue: {true_label}"
    )

    axes[0].axis("off")

    axes[1].imshow(
        cam,
        cmap="jet"
    )

    axes[1].set_title(
        "Multi-Layer LayerCAM"
    )

    axes[1].axis("off")

    axes[2].imshow(
        overlay
    )

    axes[2].set_title(
        f"{status}\n"
        f"Predicted: {predicted_label}\n"
        f"Confidence: {confidence:.3f}"
    )

    axes[2].axis("off")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=250,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

CORRECT_DIR = os.path.join(
    OUTPUT_DIR,
    "correct_examples"
)

INCORRECT_DIR = os.path.join(
    OUTPUT_DIR,
    "incorrect_examples"
)

os.makedirs(
    CORRECT_DIR,
    exist_ok=True
)

os.makedirs(
    INCORRECT_DIR,
    exist_ok=True
)


# ============================================================
# SELECT CORRECT / INCORRECT
# ============================================================

correct_df = pred_df[
    pred_df["correct"]
].copy()

incorrect_df = pred_df[
    ~pred_df["correct"]
].copy()

records = []


# ============================================================
# PROCESS EXAMPLES
# ============================================================

for diagnostic in TARGET_CLASSES:

    # --------------------------------------------------------
    # CORRECT EXAMPLE
    # --------------------------------------------------------

    examples = correct_df[
        correct_df["diagnostic"] == diagnostic
    ].head(
        N_CORRECT_PER_CLASS
    )

    for number, (_, row) in enumerate(
        examples.iterrows(),
        start=1
    ):

        print(
            f"\nCORRECT {diagnostic} #{number}"
        )

        image_tensor = load_image(
            row["image_path"]
        )

        predicted_index = int(
            row["predicted_class_index"]
        )

        predicted_label = (
            model_class_names[
                predicted_index
            ]
        )

        cam, probabilities = (
            generate_multilayer_layercam(
                image_tensor,
                predicted_index
            )
        )

        confidence = float(
            probabilities[
                predicted_index
            ]
        )

        filename = (
            f"correct_"
            f"{diagnostic.lower()}_"
            f"{number}.png"
        )

        output_path = os.path.join(
            CORRECT_DIR,
            filename
        )

        save_xai_figure(
            image_tensor,
            cam,
            row["mapped_label"],
            predicted_label,
            confidence,
            output_path,
            "CORRECT"
        )

        records.append(
            {
                "type": "correct",
                "diagnostic": diagnostic,
                "true_label": row["mapped_label"],
                "predicted_label": predicted_label,
                "confidence": confidence,
                "image_path": row["image_path"],
                "xai_path": output_path
            }
        )

        del image_tensor
        del cam
        del probabilities

        gc.collect()

    # --------------------------------------------------------
    # INCORRECT EXAMPLE
    # --------------------------------------------------------

    examples = incorrect_df[
        incorrect_df["diagnostic"] == diagnostic
    ].head(
        N_INCORRECT_PER_CLASS
    )

    for number, (_, row) in enumerate(
        examples.iterrows(),
        start=1
    ):

        print(
            f"\nINCORRECT {diagnostic} #{number}"
        )

        image_tensor = load_image(
            row["image_path"]
        )

        predicted_index = int(
            row["predicted_class_index"]
        )

        predicted_label = (
            model_class_names[
                predicted_index
            ]
        )

        # Explain the class chosen by the model.
        cam, probabilities = (
            generate_multilayer_layercam(
                image_tensor,
                predicted_index
            )
        )

        confidence = float(
            probabilities[
                predicted_index
            ]
        )

        filename = (
            f"incorrect_"
            f"{diagnostic.lower()}_"
            f"{number}.png"
        )

        output_path = os.path.join(
            INCORRECT_DIR,
            filename
        )

        save_xai_figure(
            image_tensor,
            cam,
            row["mapped_label"],
            predicted_label,
            confidence,
            output_path,
            "INCORRECT"
        )

        records.append(
            {
                "type": "incorrect",
                "diagnostic": diagnostic,
                "true_label": row["mapped_label"],
                "predicted_label": predicted_label,
                "confidence": confidence,
                "image_path": row["image_path"],
                "xai_path": output_path
            }
        )

        del image_tensor
        del cam
        del probabilities

        gc.collect()


# ============================================================
# SAVE XAI INDEX
# ============================================================

xai_df = pd.DataFrame(
    records
)

xai_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "xai_examples.csv"
    ),
    index=False
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("LayerCAM XAI COMPLETED")
print("=" * 70)

print(
    "Generated examples:",
    len(xai_df)
)

print(
    "Output directory:",
    OUTPUT_DIR
)

print("\nDone.")