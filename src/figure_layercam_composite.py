import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

XAI_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "EfficientNetV2L",
    "XAI"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "EfficientNetV2L",
    "paper_figures"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

examples = [
    ("Correct BCC", "correct_examples/correct_bcc_1.png"),
    ("Correct MEL", "correct_examples/correct_mel_1.png"),
    ("Correct SCC", "correct_examples/correct_scc_1.png"),
    ("Correct SEK", "correct_examples/correct_sek_1.png"),
    ("Incorrect BCC", "incorrect_examples/incorrect_bcc_1.png"),
    ("Incorrect MEL", "incorrect_examples/incorrect_mel_1.png"),
    ("Incorrect SCC", "incorrect_examples/incorrect_scc_1.png"),
    ("Incorrect SEK", "incorrect_examples/incorrect_sek_1.png"),
]

fig, axes = plt.subplots(
    4,
    2,
    figsize=(12, 20)
)

for ax, (title, relative_path) in zip(
    axes.flat,
    examples
):

    image_path = os.path.join(
        XAI_DIR,
        relative_path
    )

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Missing XAI image: {image_path}"
        )

    image = mpimg.imread(
        image_path
    )

    ax.imshow(image)
    ax.set_title(
        title,
        fontsize=12,
        fontweight="bold"
    )
    ax.axis("off")

fig.suptitle(
    "Multi-Layer LayerCAM Explanations on PAD-UFES-20",
    fontsize=17,
    fontweight="bold",
    y=0.995
)

plt.tight_layout(
    rect=[0, 0, 1, 0.985]
)

output_path = os.path.join(
    OUTPUT_DIR,
    "Figure8_PAD_UFES20_LayerCAM_Examples.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("=" * 70)
print("FIGURE 8 GENERATED")
print("=" * 70)
print("Output:")
print(output_path)

