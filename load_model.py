import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
import tensorflow as tf
import keras

from PIL import Image
import cv2
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_shape=(64,65,4)

model = load_model('checkpoint-16-0.42.h5')

# model.summary()
test_img = cv2.imread("Donut_10.jpg")
# print(test_img.dtype)
# test_img.imshow()
# test_img = np.reshape(test_img, ((0,) + test_img.shape))
print(np.ndim(test_img))

# print('The Answer is ', model.predict_classes(test_img))