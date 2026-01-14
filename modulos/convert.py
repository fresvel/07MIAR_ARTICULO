from tensorflow import keras
from skimage.io import imread
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

# compile=False since it is only for inference
# safe_mode=False to enable unsafe deserialization
resunet = keras.models.load_model("BestResUNet.hdf5", compile=False, safe_mode=False)
resunet.name = "Deep_Residual_UNet"
resunet.summary()

resunet.save("pretrained_model.keras")


lambdas = [l for l in resunet.layers if isinstance(l, tf.keras.layers.Lambda)]
print("Lambdas:", len(lambdas))
for l in lambdas:
    print(l.name, l.function)



for l in resunet.layers:
    if isinstance(l, tf.keras.layers.Lambda):
        print(l.name)
        print(l.get_config().keys())
        print(l.get_config())





old = tf.keras.models.load_model("pretrained_model.hdf5", compile=False)

def replace_lambda(layer):
    if isinstance(layer, tf.keras.layers.Lambda) and layer.name == "lambda":
        # Equivalente portable a: x / 255
        return tf.keras.layers.Rescaling(1./255, name="lambda")
    return layer

new = tf.keras.models.clone_model(old, clone_function=replace_lambda)
new.set_weights(old.get_weights())

# Guarda en formato Keras v3 (portable)
new.save("pretrained_model_portable.keras")
print("OK -> pretrained_model_portable.keras")
