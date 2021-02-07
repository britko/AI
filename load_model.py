import sys, os
import tensorflow as tf
import keras

from PIL import Image
import cv2
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def imgG(img_file):
    image = cv2.imread(img_file)
    image = tf.image.resize(image, [64, 64])
    image = tf.expand_dims(image, axis=0)   # the shape would be (1, 64, 64, 3)

print(imgG(r'C:\Users\pc\Desktop\고영국\개발\AI\Donut_10.jpg').shape)

model = load_model('checkpoint_v2-07-0.55-0.82.h5')

print(model.predict_classes(image))

model.summary()
test_img = cv2.imread("Donut_10.jpg")
print(test_img.dtype)
test_img.imshow()
test_img = np.reshape(test_img, ((0,) + test_img.shape))
print(np.ndim(test_img))

print('The Answer is ', model.predict_classes(test_img))