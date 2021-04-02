from tensorflow.keras import models
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

from create_DS import test_dir_glasses,train_dir, validation_dir, test_dir

image = []


path = os.path.join(test_dir, "Glasses")
print(path)
path_img = os.listdir(path)
print(path_img)
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

X = np.array(images, dtype='float').reshape(-1, 150,150,3)
X /= 255


"""
for img in path_img:
    try:
        img_path = os.path.join(test_dir_glasses, img)
        print(img_path)
        img = cv2.imread(os.path.join(test_dir_glasses, "glass_2001.jpg"))
        new_img = cv2.resize(img, (150,150))
        image.append(new_img)
    except Exception as e:
        print(str(e))

    if len(image) >= 1:
        break

"""



model = models.load_model("NN VGG16.h5")
model.summary()

predict = model.predict(X)
print(predict)
