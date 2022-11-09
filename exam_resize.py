import os
from PIL import Image

def image_resize(w, h):
    dir_name = "./exam/"
    new_dir_name = "./exam/"
    files = os.listdir(dir_name)
    for file in files:
        if file == '.DS_Store':
            continue
        photo = Image.open(os.path.join(dir_name, file))
        photo_resize = photo.resize((w, h))
        photo_resize.save(os.path.join(new_dir_name, file))

    return photo_resize

if __name__ == '__main__':
    exam_dataset = image_resize(224, 224)