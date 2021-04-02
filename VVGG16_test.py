#from tensorflow.keras import models
import cv2
import os
import matplotlib.pyplot as plt

from create_DS import test_dir_glasses
path_img = os.listdir(test_dir_glasses)
image = []

for img in test_dir_glasses:
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



plt.imshow(new_img)
plt.show()

#model = models.load_model("NN VGG16.h5")
#model.summary()





model.predict()
