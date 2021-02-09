import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing import image

# 분류 가능하게 이미지 가공
img1 = image.load_img(r'C:\Users\pc\Desktop\고영국\개발\AI\test_img\None_9.jpg', target_size=(64, 64))
img = image.img_to_array(img1)
img = img/255
img = np.expand_dims(img, axis=0)

# print(img.shape)

# 모델 불러오기
model = load_model('checkpoint_v2-07-0.55-0.82.h5')

# 분류!!!
print('The Answer is ', model.predict_classes(img))