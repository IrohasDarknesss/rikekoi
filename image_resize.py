import os
from PIL import Image

def image_resize(w, h, name):
    dir_name = "./datasets/{}/".format(name)
    new_dir_name = "./datasets/{}/".format(name)
    files = os.listdir(dir_name)
    for file in files:
        if file == '.DS_Store':
            continue
        photo = Image.open(os.path.join(dir_name, file))
        photo_resize = photo.resize((w, h))
        photo_resize.save(os.path.join(new_dir_name, file))

    return photo_resize

if __name__ == '__main__':
    himuro_dataset = image_resize(224, 224, 'himuro')
    ibarada_dataset = image_resize(224, 224, 'ibarada')
    kanade_dataset = image_resize(224, 224, 'kanade')
    kosuke_dataset = image_resize(224, 224, 'kosuke')
    yukimura_dataset = image_resize(224, 224, 'yukimura')