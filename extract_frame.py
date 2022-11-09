import cv2
import numpy as np
 
 # m_slice：path = file_path、save_path、steo_num、extention）
def m_slice(path, dir, step, extension):
    movie = cv2.VideoCapture(path)                  # load mp4
    Fs = int(movie.get(cv2.CAP_PROP_FRAME_COUNT))   # calc all frame
    path_head = dir + 'rikekoi_'                       # image header
    ext_index = np.arange(0, Fs, step)              # extract index
 
    for i in range(Fs - 1):                         # roop for frame num
        flag, frame = movie.read()                  # load 1 frame from movie
        check = i == ext_index                      # check
        
        # When flag=True, only execute. 
        if flag == True:
            # If the i-th frame is to extract a still image, name and save the file.
            if True in check:
                # File names should be sequentially numbered when sorted by name later in the folder.
                if i < 10:
                    path_out = path_head + '0000' + str(i) + extension
                elif i < 100:
                    path_out = path_head + '000' + str(i) + extension
                elif i < 1000:
                    path_out = path_head + '00' + str(i) + extension
                elif i < 10000:
                    path_out = path_head + '0' + str(i) + extension
                else:
                    path_out = path_head + str(i) + extension
                cv2.imwrite(path_out, frame)
            # If the i-th frame is one from which no still image is extracted, no processing is done.
            else:
                pass
        else:
            pass
    return
 
m_slice('./video/rikekoi3.mp4', './data/', 35, '.jpg')