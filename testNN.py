import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras import models
import matplotlib.pyplot as plt
import os
from create_DS import test_dir_tables

test_dir_tables_img1 = os.path.join(test_dir_tables, "table_3201.jpg")

def load_image(img_dir, show=True):

    img = image.load_img(img_dir, target_size=(150, 150))
    image_tensor = image.img_to_array(img)                    # (height, width, channels)
    print(image_tensor.shape)
    image_tensor = np.expand_dims(image_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    print(image_tensor.shape)
    image_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(image_tensor[0])
        plt.axis('off')
        plt.show()

    return image_tensor


load_image(test_dir_tables_img1)
