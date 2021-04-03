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

def get_data(Categories, dir):
    Dataset = []
    for Categorie in Categories:
        path = os.path.join(dir, Categorie)

        path_img = os.listdir(path)
        label = Categories.index(Categorie)


        for img in path_img:

            img_array = cv2.imread(os.path.join(path,img))
            new_img_array = cv2.resize(img_array,(150,150))
            #plt.clf()
            #plt.imshow(new_img_array)
            #plt.show()
            Dataset.append([new_img_array, label])

    return Dataset


data = get_data(Categories,train_dir)
validation_data = get_data(Categories2,validation_dir)


def get_x_y(data):
    x,y = [],[]
    for feature, label in data:
        x.append(feature)
        y.append(label)

    return x,y

x_train,y_train = get_x_y(data)
x_validation,y_validation = get_x_y(validation_data)



X = np.array(x_train, dtype='float').reshape(-1,150,150,3)
X /= 255
y = np.array(y_train)

X_val = np.array(x_validation, dtype='float').reshape(-1,150,150,3)
X_val /= 255
y_val = np.array(y_validation)




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

history = model.fit(X,y, batch_size=32, epochs=15, validation_data=(X_val, y_val))
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
