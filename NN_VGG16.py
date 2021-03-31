from tensorflow.keras.applications import VGG16
from tensorflow.keras import utils
from tensorflow.keras.applications import VGG16
from tensorflow.keras import models, layers, optimizers

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_io as tfio

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

    count = 0

    for img in path_img:

        img_array = cv2.imread(os.path.join(path,img))
        new_img_array = cv2.resize(img_array,(150,150))
        #plt.clf()
        #plt.imshow(new_img_array)
        #plt.show()
        data.append([new_img_array, label])




x = []
y = []
for feature, label in data:
    print(feature.shape)
    x.append(feature)
    y.append(label)

X = np.array(x).reshape(-1,150,150,3)
X //= 255

y = np.array(y)





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



model.compile(loss="binary_crossentropy", optimizer=optimizers.RMSprop(lr=2e-5), metrics=['acc'])

history = model.fit(X,y, batch_size=32, epochs=5)
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
