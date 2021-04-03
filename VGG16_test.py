from tensorflow.keras import models
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

from create_DS import test_dir_glasses,train_dir, validation_dir, test_dir



path = os.path.join(test_dir, "Glasses")
print(path)
path_img = os.listdir(path)
images = []
count = 0
for img in path_img:
    img_array = cv2.imread(os.path.join(path,img))
    new_img_array = cv2.resize(img_array,(150,150))
    images.append(new_img_array)
    #plt.imshow(new_img_array)
    #plt.show()
    if count > 12:
        break
    count += 1

path = os.path.join(test_dir, "Tables")
print(path)
path_img = os.listdir(path)

count = 0
for img in path_img:
    img_array = cv2.imread(os.path.join(path,img))
    new_img_array = cv2.resize(img_array,(150,150))
    images.append(new_img_array)
    #plt.imshow(new_img_array)
    #plt.show()
    if count > 12:
        break
    count += 1

model = models.load_model("NN VGG16.h5")
model.summary()

def prediction(images):
    n_glasses = 0
    n_tables = 0
    for image in images:
        X = np.array(image, dtype='float').reshape(-1, 150,150,3)
        X /= 255
        predict = model.predict(X)
        if predict <= 0.5:
            print("glass")
            n_glasses += 1

        else:
            print("table")
            n_tables += 1

    print(n_glasses, n_tables)

prediction(images)
