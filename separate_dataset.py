import shutil
import glob
import random
import os


def separate_dataset(sep, name):
    path_list = glob.glob('./datasets/train/{}/*jpg'.format(name))
    file_list = path_list
    list_len = len(file_list)
    n = int(list_len*sep)

    z = random.sample(file_list, list_len)
    train_list = z[:n]
    test_list = z[n:]

    val_file = './datasets/test/{}/'.format(name)

    print (len(file_list))
    print (len(train_list))
    print (len(test_list))

    for test_path in test_list:
        print(test_path)
        shutil.move(test_path, val_file.format(name)+os.path.basename(test_path))
    return [train_list,test_list]

if __name__ == '__main__':
   himuro_dataset = separate_dataset(0.7,'himuro')
   ibarada_dataset = separate_dataset(0.7, 'ibarada')
   kanade_dataset = separate_dataset(0.7, 'kanade')
   kosuke_dataset = separate_dataset(0.7, 'kosuke')
   yukimura_dataset = separate_dataset(0.7, 'yukimura')