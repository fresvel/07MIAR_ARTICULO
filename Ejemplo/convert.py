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
