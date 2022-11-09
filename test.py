import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob
from torchsummary import summary
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

#Function to return a square around the person in the image, with the name appended.
def detect_who(IMAGE_PATH,model):
    image=cv2.imread(IMAGE_PATH)
    if image is None:
        print("Not open:")
    image_gs=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    cascade=cv2.CascadeClassifier("./cascade/lbpcascade_animeface.xml")
    face_list=cascade.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=2,minSize=(64,64))

    count=0
    if len(face_list)>0:
        for rect in face_list:
            count+=1
            x,y,width,height=rect
            print(x,y,width,height)
            image_face=image[y:y+height,x:x+width]
            if image_face.shape[0]<64:
                continue
            image_face = cv2.resize(image_face,(64,64))
            transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            image_face = transform(image_face)
            image_face = image_face.view(1,3,64,64)

            #print(image_face.shape)

            output=model(image_face)

            member_label=output.argmax(dim=1, keepdim=True)#The element number with the largest element value in the model output is the member label.
            name = label2name(member_label)#Identify people from labels.

            print(output)
            cv2.rectangle(image, (x,y), (x+width,y+height), (255, 0, 0), thickness=3)#Quadrangle drawing
            cv2.putText(image,name,(x,y+height+20),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0),2)#Personal name description
    else:
        print("no face")

    return image

#Function to identify names from labels. If classification is performed on a different dataset, this part needs to be changed.
def label2name(member_label):
    if member_label==0:
        name='himuro'
    elif member_label==1:
        name='ibarada'
    elif member_label==2:
        name='kanade'
    elif member_label==3:
        name='kosuke'
    elif member_label==4:
        name='yukimura'
    return name


def main():
    #Loading of saved models can be done with the following three lines.
    # resnet model loading.
    resnet = models.resnet18(pretrained=True)
    # Replaced the original 1000-class classification with a 5-class classification
    resnet.fc = nn.Linear(512,5)
    # print model
    print(resnet)
    model = resnet
    model.load_state_dict(torch.load('./model/rikekoi_09.pth'))
    model.eval()

    path_list = glob.glob('./exam/*')
    print(path_list)

    out_dir='./result'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for path in path_list:
        image_name=path.split('/')[-1]
        Who = detect_who(path,model)
        SAVE_PATH=os.path.join(out_dir+'/'+image_name)
        cv2.imwrite(SAVE_PATH,Who)

if __name__ == "__main__":
    main()