import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.image import imread

from sklearn.metrics import classification_report,confusion_matrix

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

import warnings
warnings.filterwarnings('ignore')

b_d=os.listdir(r'C:/Users/pc/Jupyter Notebook/WaferMap/balanced')
for i in b_d:
   print( i,len(os.listdir(r'C:/Users/pc/Jupyter Notebook/WaferMap/balanced/'+i)))
print("==================================================================")
test_d=os.listdir(r'C:/Users/pc/Jupyter Notebook/WaferMap/imbalanced')
for i in test_d:
   print( i,len(os.listdir(r'C:/Users/pc/Jupyter Notebook/WaferMap/imbalanced/'+i)))

b_path=r'C:/Users/pc/Jupyter Notebook/WaferMap/balanced/'
sample_wafer=[]
for i in b_d:
    sample_wafer.append(b_path+i+'/'+os.listdir(b_path+i+'/')[0])
sample_wafer

# plt.figure(figsize=(24,12))
f, axarr = plt.subplots(3,3,figsize=(24,12))
m=0
for i in range(3):
    for j in range(3):
        axarr[i,j].imshow(imread(sample_wafer[m]))
        axarr[i,j].set_title(os.path.basename(sample_wafer[m])) 
        m+=1

def dimension(path,dim1,dim2):
    for image_filename in os.listdir(path): 
        image=imread(path+image_filename)
        d1,d2,channels=image.shape
        dim1.append(d1)
        dim2.append(d2)
#         print(channels)
    return dim1,dim2

loc_dim1=[]
loc_dim2=[]
loc_dim1,loc_dim2=dimension(r'C:/Users/pc/Jupyter Notebook/WaferMap/balanced/Loc/',loc_dim1,loc_dim2)

edgeRing_dim1=[]
edgeRing_dim2=[]
edgeRing_dim1,edgeRing_dim2=dimension(r'C:/Users/pc/Jupyter Notebook/WaferMap/balanced/Edge-ring/',edgeRing_dim1,edgeRing_dim2)

edgeLoc_dim1=[]
edgeLoc_dim2=[]
edgeLoc_dim1,edgeLoc_dim2=dimension(r'C:/Users/pc/Jupyter Notebook/WaferMap/balanced/Edge-loc/',edgeLoc_dim1,edgeLoc_dim2)

center_dim1=[]
center_dim2=[]
center_dim1,center_dim2=dimension(r'C:/Users/pc/Jupyter Notebook/WaferMap/balanced/Center/',center_dim1,center_dim2)

random_dim1=[]
random_dim2=[]
random_dim1,random_dim2=dimension(r'C:/Users/pc/Jupyter Notebook/WaferMap/balanced/Random/',random_dim1,random_dim2)

scratch_dim1=[]
scratch_dim2=[]
scratch_dim1,scratch_dim2=dimension(r'C:/Users/pc/Jupyter Notebook/WaferMap/balanced/Scratch/',scratch_dim1,scratch_dim2)

nearFull_dim1=[]
nearFull_dim2=[]
nearFull_dim1,nearFull_dim2=dimension(r'C:/Users/pc/Jupyter Notebook/WaferMap/balanced/Near-Full/',nearFull_dim1,nearFull_dim2)

donut_dim1=[]
donut_dim2=[]
donut_dim1,donut_dim2=dimension(r'C:/Users/pc/Jupyter Notebook/WaferMap/balanced/Donut/',donut_dim1,donut_dim2)

none_dim1=[]
none_dim2=[]
none_dim1,none_dim2=dimension(r'C:/Users/pc/Jupyter Notebook/WaferMap/balanced/None/',donut_dim1,donut_dim2)

image_gen = ImageDataGenerator(rotation_range=20, # rotate the image 20 degrees
                               width_shift_range=0.10, # Shift the pic width by a max of 5%
                               height_shift_range=0.10, # Shift the pic height by a max of 5%
                               rescale=1/255, # Rescale the image by normalzing it.
                               shear_range=0.1, # Shear means cutting away part of the image (max 10%)
                               zoom_range=0.1, # Zoom in by 10% max
                               horizontal_flip=True, # Allo horizontal flipping
                               fill_mode='nearest' # Fill in missing pixels with the nearest filled value
                              )


train_path=r'C:/Users/pc/Jupyter Notebook/WaferMap/balanced'
test_path=r'C:/Users/pc/Jupyter Notebook/WaferMap/imbalanced'
# image_gen.flow_from_directory(train_path)
# image_gen.flow_from_directory(test_path)

batch_size = 16
img_shape=(64,65,4)

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=img_shape, activation='relu',))
model.add(MaxPool2D(pool_size=(2, 2)))
## FIRST SET OF LAYERS
# CONVOLUTIONAL LAYER
# POOLING LAYER
model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=img_shape, activation='relu',))
model.add(MaxPool2D(pool_size=(2, 2)))

## SECOND SET OF LAYERS
# CONVOLUTIONAL LAYER
# POOLING LAYER
model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=img_shape, activation='relu',))
model.add(MaxPool2D(pool_size=(2, 2)))

# model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=img_shape, activation='relu',))
# model.add(MaxPool2D(pool_size=(2, 2)))


# FLATTEN IMAGES FROM 64 by 65 to 4160 BEFORE FINAL LAYER
model.add(Flatten())

# 256 NEURONS IN DENSE HIDDEN LAYER (YOU CAN CHANGE THIS NUMBER OF NEURONS)
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
# model.add(Dense(128, activation='sigmoid'))
# model.add(Dropout(0.5))
# LAST LAYER IS THE CLASSIFIER, THUS 9 POSSIBLE CLASSES
model.add(Dense(9, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


model.summary()

#####################################################################################################################
import tensorflow as tf
from keras.callbacks import ModelCheckpoint

'''my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2),
    tf.keras.callbacks.ModelCheckpoint(filepath=r'C:/Users/pc/Jupyter Notebook/model.{epoch:02d}-{val_loss:.2f}.h5'),
    tf.keras.callbacks.TensorBoard(log_dir=r'C:/Users/pc/Jupyter Notebook/logs'),
]'''

filename = r'C:/Users/pc/Jupyter Notebook/savemodel/checkpoint-{epoch:02d}-{val_loss:.2f}.h5'
checkpoint = ModelCheckpoint(filename,             # file명을 지정합니다
                             monitor='val_loss',   # val_loss 값이 개선되었을때 호출됩니다
                             verbose=1,            # 로그를 출력합니다
                             save_best_only=True,  # 가장 best 값만 저장합니다
                             mode='auto'           # auto는 알아서 best를 찾습니다. min/max
                            )

train_image_gen = image_gen.flow_from_directory(train_path,
                                               target_size=img_shape[:2],
                                                color_mode='rgba',
                                               batch_size=batch_size,
                                               class_mode='categorical')

test_image_gen = image_gen.flow_from_directory(test_path,
                                               target_size=img_shape[:2],
                                                color_mode='rgba',
                                               batch_size=batch_size,
                                               class_mode='categorical',
                                              shuffle=False)


results = model.fit(
    train_image_gen,
    validation_data=test_image_gen,
    epochs=20,
    callbacks=[checkpoint]
)