from tensorflow.keras.applications import VGG16
from tensorflow.keras import models, layers
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import matplotlib.pyplot as plt

from create_DS import train_dir, validation_dir

conv_base = VGG16(weights="imagenet", include_top=False, input_shape=(150,150,3))
conv_base.trainable

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


test_datagen = ImageDataGenerator(rescale=1./255)
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip = True)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150,150), batch_size=32, class_mode='binary', color_mode='rgba')
validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(150,150), batch_size=32, class_mode='binary', color_mode='rgba') #'rgba' because we have images with transparency

for data_batch, labels_batch in train_generator:
    print("shape of a data batch",data_batch.shape)

    break
