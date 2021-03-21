from keras.tensorflow import models
from keras.tensorflow import layers
from keras.tensorflow import metrics
from keras.tensorflow.applications import VGG16

import matplotlib.pyplot as plt

from create_DS import 

#conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150,150,3))


test = ImageDataGenerator(rescale=1./255) #Rescale is a value by which we will multiply the data
#before any other processing. Our original images consist in RGB coefficients
#in the 0-255, but such values would be too high for our model to process
#(given a typical learning rate), so we target values between 0 and 1
#instead by scaling with a 1/255
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip = True)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(200,200), batch_size=32, class_mode='binary')
validation_generator = test_datagen.flow_from_directory()
