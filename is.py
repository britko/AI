import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint

rootPath = 'C:/Users/pc/Jupyter Notebook/WaferMap'

imageGenerator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[.2, .2],
    horizontal_flip=True,
    validation_split=.1,
    fill_mode='nearest'
)

trainGen = imageGenerator.flow_from_directory(
    'C:/Users/pc/Jupyter Notebook/WaferMap/imbalanced',
    batch_size=16,
    target_size=(64, 64),
    color_mode='rgb',
    subset='training',
    class_mode='categorical'
)

validationGen = imageGenerator.flow_from_directory(
    'C:/Users/pc/Jupyter Notebook/WaferMap/balanced',
    batch_size=16,
    target_size=(64, 64),
    color_mode='rgb',
    subset='validation',
    class_mode='categorical',
    shuffle=False
)


print(validationGen.image_shape)