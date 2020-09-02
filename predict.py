import cv2
import sys
import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python import keras as K


def face_cut(img):
    face_cascade_path = 'opencv/haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(img_gray, 1.1, 3)
    if len(face) > 0:
        for x, y, w, h in face:
            img_cut = img[y:y+h, x:x+w]
            resized_img = cv2.resize(img_cut, (200, 200))
            reshaped_img = resized_img.reshape([1, 200, 200, -1])
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
            return reshaped_img, img
    else:
        print('No Face.')
        sys.exit()


model = tf.keras.models.load_model('saved_model/yoji_0.h5')
img = cv2.imread('predict_img/4.jpg')

label = {0 : '野田洋次郎', 1 : '吉田沙保里', 2 : '羽生結弦'}

face ,img = face_cut(img)

predict = model.predict(face)
num = np.argmax(predict)

print('x----------x')
print(label[num])
print('x----------x')

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.show()
