from tensorflow.keras.applications import VGG16
from tensorflow.keras import models, layers
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import matplotlib.pyplot as plt

from create_DS import train_dir, validation_dir

conv_base = VGG16(weights="imagenet", include_top=False, input_shape=(150,150,3))
step_size_train = train_generator.n//train_generator.batch_size
step_size_valid = validation_generator.n//validation_generator.batch_size


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
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip = True)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150,150), batch_size=32, class_mode='binary', color_mode='rgba')
validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(150,150), batch_size=32, class_mode='binary', color_mode='rgba') #'rgba' because we have images with transparency


for data_batch, labels_batch in train_generator:
    print("shape of a data batch",data_batch.shape)

    break

model.compile(loss="binary_crossentropy", optimizer=optimizers.RMSprop(lr=2e-5), metrics=['acc'])

history = model.fit(train_generator, steps_per_epoch=100, epochs=30,validation_data=validation_generator, validation_steps=50)
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
