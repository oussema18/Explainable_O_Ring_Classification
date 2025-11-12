import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from lime import lime_image
from skimage.segmentation import mark_boundaries
from skimage.color import label2rgb

# ---- Model ----
model = InceptionV3(weights="imagenet", include_top=True)  # outputs softmax probs
IMAGE_SIZE = 299

# ---- Load raw RGB (0..255), no preprocessing here ----
def load_raw(paths):
    imgs = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        imgs.append(np.array(img, dtype=np.uint8))  # uint8 0..255
    return np.stack(imgs, axis=0)                   # (N,299,299,3)

# ---- LIME-compatible predictor (does preprocessing inside) ----
def predict_fn(np_imgs):
    # np_imgs: (N,299,299,3) uint8 or float32 0..255
    x = np_imgs.astype(np.float32)
    x = preprocess_input(x)              # -> [-1,1] as InceptionV3 expects
    preds = model(x, training=False).numpy()   # (N,1000) softmax probs
    return preds

# ---------------- Demo ----------------
raw_images = load_raw(['dogs.jpg'])      # raw for display & LIME
probs = predict_fn(raw_images)

# Top-5 labels
top5 = decode_predictions(probs, top=5)[0]
for (_, label, p) in top5:
    print(f"{label:20s} {p:.4f}")

# Show the (raw) image
plt.imshow(raw_images[0])
plt.axis('off')
plt.tight_layout()
plt.show()

# ---- LIME ----
explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(
    raw_images[0],            # pass RAW image
    predict_fn,               # batch function returning probs
    top_labels=5,
    hide_color=0,
    num_samples=1000
)

# Pick the top predicted class (or any class index you want)
top_class = int(np.argmax(probs[0]))

# Get LIME mask and visualize
temp, mask = explanation.get_image_and_mask(
    top_class,
    positive_only=False,
    num_features=10,
    hide_rest=False
)

plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))

# Also save to file if GUI backends are flaky:
plt.imsave("lime_explanation.png", mark_boundaries(temp, mask))
print("Saved LIME figure to lime_explanation.png")
