import os
import json
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import shap
# https://shap.readthedocs.io/en/latest/example_notebooks/image_examples/image_classification/Explain%20ResNet50%20using%20the%20Partition%20explainer.html
# Configure TensorFlow
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)

# load pre-trained model and data
model = ResNet50(weights="imagenet")
X, y = shap.datasets.imagenet50()

# getting ImageNet 1000 class names
url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
with open(shap.datasets.cache(url)) as file:
    class_names = [v[1] for v in json.load(file).values()]

def f(x):
    tmp = x.copy()
    preprocess_input(tmp)
    return model(tmp)

def inpaint_telea():
    # define a masker that is used to mask out partitions of the input image.
    # This simulates “removing” parts of the image to measure feature importance.
    masker = shap.maskers.Image("inpaint_telea", X[0].shape)

    # create an explainer with model and image masker
    # explainer(X[1:3], …) Explains 2 images (X[1], X[2]).
    explainer = shap.Explainer(f, masker, output_names=class_names)

    # here we explain two images using 500 evaluations of the underlying model to estimate the SHAP values
    # outputs=shap.Explanation.argsort.flip[:4]: get SHAP values for the top 4 predicted classes.
    shap_values = explainer(X[1:3], max_evals=100, batch_size=5, outputs=shap.Explanation.argsort.flip[:4])
    return shap_values

def blur():
   # define a masker that is used to mask out partitions of the input image.
    masker_blur = shap.maskers.Image("blur(128,128)", X[0].shape)

    # create an explainer with model and image masker
    explainer_blur = shap.Explainer(f, masker_blur, output_names=class_names)

    # here we explain two images using 500 evaluations of the underlying model to estimate the SHAP values
    shap_values_fine = explainer_blur(X[1:4], max_evals=300, batch_size=5, outputs=shap.Explanation.argsort.flip[:4])
    return shap_values_fine
shap_values = blur()
#shap_values = inpaint_telea()
shap.image_plot(shap_values)
'''
In the first example, given bird image is classified as an American Egret with next probable classes being a Crane, Heron and Flamingo. It is the “bump” over the 
bird's neck that causes it to be classified as an American Egret vs a Crane, Heron or a Flamingo. You can see the neck region of the bird appropriately highlighted 
in red super pixels.

In the second example, it is the shape of the boat which causes it to be classified as a speedboat instead of a fountain, lifeboat or snowplow (appropriately highlighted
in red super pixels).
'''