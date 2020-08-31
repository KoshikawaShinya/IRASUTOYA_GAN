import cv2
import matplotlib.pyplot as plt
import numpy as np

# cv2ではRGBがBGRの順で読み込まれるため色がおかしくなる
img_bgr = cv2.imread('imgs/original/yoji1/1.20190828-oyt1i50065-1.jpg')
# BGRをRGB
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
# BGRをHSV(Hue:色相,Saturation:彩度,Value:明度)
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
# RGBをグレースケール
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
# RGBをLAB(L:輝度,A:赤-緑成分,B:黄-青成分)
img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
# RGBを二値化
retval, img_two = cv2.threshold(img_rgb, 100, 128, cv2.THRESH_BINARY)
# ぼかし
img_blur = cv2.GaussianBlur(img_rgb, (15, 15), 0)

# サイズ変更
size = img_rgb.shape
new_img = img_rgb[size[0] // 3 : , : size[1] // 2]
print(new_img.shape)

# 各次元の画素数を変更(この場合各次元を1/10にしている)
resized_img = cv2.resize(img_rgb, (img_rgb.shape[1] // 10, img_rgb.shape[0] // 10))
print(resized_img.shape)

# 回転
# cv2.getRotationMatrix2D(回転の中心(画像の中心), 回転角度, 拡大する倍率) 変換の「指示」を設定するイメージ
mat = cv2.getRotationMatrix2D(tuple(np.array(img_rgb.shape[:2]) / 2), 45, 0.8)
# cv2.warpAffine(返還対象, 変換行列, 出力する画像のサイズ) 
rolled_img = cv2.warpAffine(img_rgb, mat, img_rgb.shape[:2])

# 反転
flip_img = cv2.flip(img_rgb, 1)

# ノイズの除去
renoized_img = cv2.fastNlMeansDenoisingColored(img_rgb)

# 輪郭抽出
retval, thresh = cv2.threshold(img_gray, 88, 255, 0)
img, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
result_img = cv2.drawContours(img, contours, -1, (0, 0, 255), 3)


print(type(img_bgr))
print(img_bgr.shape)
plt.imshow(result_img)
plt.show()