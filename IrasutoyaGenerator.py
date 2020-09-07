import keras as K
import cv2
import matplotlib.pyplot as plt
import numpy as np


seed = 502

z_dim = 100

model = K.models.load_model('saved_model/mizumashi_model.h5')

#np.random.seed(seed=seed)
z = np.random.normal(0, 1, (1, z_dim))

img = model.predict(z)
img = np.reshape(img, (64, 64, 3))
print(img.shape)

img = 0.5 * img + 0.5

plt.imsave('generate_img/irasuto_2.jpg', img)
plt.imshow(img)
plt.show()
