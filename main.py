from keras.tensorflow import models
from keras.tensorflow import layers
from keras.tensorflow import metrics
from keras.tensorflow.applications import VGG16

conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150,150,3))

train_ = ImageDataGenerator(rescale=1./255)
