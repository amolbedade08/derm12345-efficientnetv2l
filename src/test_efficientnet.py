import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2L

print("=" * 60)
print("TENSORFLOW / GPU TEST")
print("=" * 60)

print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))

# Allow TensorFlow to grow GPU memory as needed
gpus = tf.config.list_physical_devices("GPU")

if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print("Memory growth error:", e)

print("\nLoading EfficientNetV2-L...")

model = EfficientNetV2L(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

print("EfficientNetV2-L loaded successfully!")

# Very small test batch
x = tf.random.uniform(
    shape=(1, 224, 224, 3)
)

print("\nRunning one forward pass...")

y = model(x, training=False)

print("Output shape:", y.shape)

print("\nSUCCESS: EfficientNetV2-L forward pass completed.")