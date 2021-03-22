from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import metrics, optimizers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True #Fix an issue with PIL which reads truncated files


import matplotlib.pyplot as plt

from create_DS import train_dir, validation_dir

#conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150,150,3))


test_datagen = ImageDataGenerator(rescale=1./255) #Rescale is a value by which we will multiply the data
#before any other processing. Our original images consist in RGB coefficients
#in the 0-255, but such values would be too high for our model to process
#(given a typical learning rate), so we target values between 0 and 1
#instead by scaling with a 1/255
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip = True)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150,150), batch_size=32, class_mode='binary', color_mode='rgba')
validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(150,150), batch_size=32, class_mode='binary', color_mode='rgba') #'rgba' because we have images with transparency

step_size_train = train_generator.n//train_generator.batch_size
step_size_valid = validation_generator.n//validation_generator.batch_size

model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])
history = model.fit(train_generator, steps_per_epoch=step_size_train,epochs=20,validation_data=validation_generator, validation_steps=step_size_valid)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(acc)+1)

plt.plot(epochs,acc,'bo', label='Training acc')
plt.plot(epochs, val_acc,'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
