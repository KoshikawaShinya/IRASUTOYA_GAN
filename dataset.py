import cv2
import glob
import numpy as np


label = 0
datas = []
labels = []

losses = []
accuracies = []
iteration_checkpoints = []

# 画像をテンソル化
for fold_path in glob.glob('imgs/resized_1/*'):
    imgs = glob.glob(fold_path + '/*')

    for img_path in imgs:
        img = cv2.imread(img_path)
        #img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        datas.append(img)
        labels.append(label)
    
    label += 1

x_train = np.array(datas)
labels = np.array(labels)
#print(image_labels[900])
#plt.imshow(image_datas[900])
#plt.show()

x_train = x_train / 127.5 - 1.0

np.savez('datasets/noda.npz', X_train=x_train, Y_train=labels)