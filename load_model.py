import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
import tensorflow as tf
import keras

from PIL import Image
import cv2
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_shape=(64,65,4)

model = load_model('checkpoint-16-0.42.h5')

# model.summary()
# test_img = cv2.imread("Donut_10.jpg")
# print(test_img.dtype)
# test_img.imshow()
# test_img = np.reshape(test_img, ((0,) + test_img.shape))
# print(np.ndim(test_img))

# print('The Answer is ', model.predict_classes(test_img))

test_image_gen = image_gen.flow_from_directory(test_path,
                                               target_size=img_shape[:2],
                                                color_mode='rgba',
                                               batch_size=batch_size,
                                               class_mode='categorical',
                                              shuffle=False)


test_image=cv2.imread("Donut_10.jpg")



eval_image = image.load_img(test_image,target_size=img_shape,color_mode='rgba')
eval_image = image.img_to_array(eval_image)
eval_image = np.expand_dims(eval_image, axis=0)
l=model.predict(eval_image)
keys=list(test_image_gen.class_indices.keys())
print('wafer defect classifed as '+ str(keys[l.argmax()]))