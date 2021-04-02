from tensorflow.keras import models

from create_DS import test_dir_glasses


model = models.load_model("NN VGG16.h5")
model.summary()



model.predict()
