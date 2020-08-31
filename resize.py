import cv2
import glob # ファイル読み込みで使用
import os # フォルダ作成で使用


for fold_path in glob.glob('imgs/mizumashi/*'):
    count = 0

    # 画像全部のディレクトリリスト
    imgs = glob.glob(fold_path + '/*')
    # 顔切り取り後の、画像保存先のフォルダ名
    save_path = fold_path.replace('mizumashi','resized_1')
    
    # 保存先のフォルダがなかったら、フォルダ作成
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 画像ごとに処理
    for i, img_path in enumerate(imgs,1):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        print(img_path)
        resized_img = cv2.resize(img, (128, 128))
        count += 1
        save_img_path = save_path + '/' + str(count) + '.jpg'
        cv2.imwrite(save_img_path, resized_img)
        
        