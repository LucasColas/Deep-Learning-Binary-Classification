"""
from tensorflow.keras.applications import VGG16
from tensorflow.keras import utils
from tensorflow.keras import models, layers, optimizers

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_io as tfio
"""
import numpy as np
import cv2
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os

import matplotlib.pyplot as plt

from create_DS import train_dir, validation_dir

Categories = ["Glass", "Tables"]
Categories2 = ["Glasses", "Tables"]


data = []

for Categorie in Categories:
    path = os.path.join(train_dir, Categorie)

    path_img = os.listdir(path)
    label = Categories.index(Categorie)

    for img in path_img:

        img_array = cv2.imread(os.path.join(path,img))
        data.append([img_array, label])



x = []
y = []
for feature, label in data:
    print(feature.shape)
    x.append(feature)
    y.append(label)



#X = np.array(x).reshape(-1,)



"""
conv_base = VGG16(weights="imagenet", include_top=False, input_shape=(150,150,3))



model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
print(model.summary())
print("trainable weights", len(model.trainable_weights))
conv_base.trainable = False #Layers with weights of the ConvNet not updated
print("trainable weights : ", len(model.trainable_weights))


test_datagen = ImageDataGenerator(rescale=1./255)
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip = True, fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150,150), batch_size=32, class_mode='binary', color_mode='rgba')
validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(150,150), batch_size=32, class_mode='binary', color_mode='rgba') #'rgba' because we have images with transparency

step_size_train = train_generator.n//train_generator.batch_size
step_size_valid = validation_generator.n//validation_generator.batch_size
print("step_size_train",step_size_train)
print("step_size_valid",step_size_valid)

for data_batch, labels_batch in train_generator:
    print("shape of a data batch",data_batch.shape)

    break





train = tfio.experimental.color.rgba_to_rgb(train_generator)
validation = tfio.experimental.color.rgba_to_rgb(validation_generator)
for data_batch, labels_batch in train:
    print("shape of a data batch after rgb",data_batch.shape)

    break

"""

"""
model.compile(loss="binary_crossentropy", optimizer=optimizers.RMSprop(lr=2e-5), metrics=['acc'])

history = model.fit(train, steps_per_epoch=32, epochs=15,validation_data=validation, validation_steps=16)
model.save("NN VGG16.h5")

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
"""
