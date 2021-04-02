from tensorflow.keras import models
import cv2
import os
import matplotlib.pyplot as plt

from create_DS import test_dir_glasses



img = cv2.imread(os.path.join(test_dir_glasses, "glass_2001.jpg"))
new_img = cv2.resize(img, (150,150))
plt.imshow(new_img)
plt.show()

model = models.load_model("NN VGG16.h5")
model.summary()





model.predict()
