from tensorflow.keras import VGG16


conv_base = VGG16(weights="imagenet", include_top=False, input_shape=(150,150,4))

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import matplotlib.pyplot as plt

from create_DS import train_dir, validation_dir
