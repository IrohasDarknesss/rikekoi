import glob

def check_length(name):
    path = glob.glob('./datasets/train/{}/*.jpg'.format(name))
    print(len(path))

if __name__ == '__main__':
    himuro_dataset = check_length('himuro')
    ibarada_dataset = check_length('ibarada')
    kanade_dataset = check_length('kanade')
    kosuke_dataset = check_length('kosuke')
    yukimura_dataset = check_length('yukimura')