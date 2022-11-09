from PIL import Image
from torchvision import transforms
import glob

def data_augumentation(name):
    
    path = glob.glob('./datasets/train/{}/*jpg'.format(name))


    # オーグメンテーション
    transform = transforms.Compose([

        # ここに処理を記載していく
        transforms.Grayscale(),
        transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=[-0.5, 0.5]),
        transforms.RandomInvert(p=1.0)
    ])
    i = 1
    for p in path:
        # 画像の読み込み
        img = Image.open(p)

        # オーグメンテーションの実行
        img = transform(img)

        # 編集した画像を保存
        img.save('./datasets/train/{}/rekekoiAug_'.format(name)+ str(i) + '.jpg')

        i += 1

    return img

if __name__ == '__main__':
    himuro_dataset = data_augumentation('himuro')
    ibarada_dataset = data_augumentation('ibarada')
    kanade_dataset = data_augumentation('kanade')
    kosuke_dataset = data_augumentation('kosuke')
    yukimura_dataset = data_augumentation('yukimura')