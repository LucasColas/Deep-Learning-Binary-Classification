import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

def load_image(img_dir, show=False):

    img = image.load_img(img_path, target_size=(150, 150))
    images_tensor = image.img_to_array(img)                    # (height, width, channels)
    images_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])
        plt.axis('off')
        plt.show()

    return img_tensor
