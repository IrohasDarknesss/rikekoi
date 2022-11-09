from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torchvision.models as models
import torch
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np

data_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

# dir
base_path = "./datasets"
# train/test dataset
train_dataset = datasets.ImageFolder(root= base_path + "/train", transform=data_transform)
test_dataset = datasets.ImageFolder(root=base_path + '/test', transform=data_transform)

# batch size
mini_batch_size = 8

# Create Loader
train_loader = DataLoader(train_dataset, batch_size=mini_batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=mini_batch_size, shuffle=False)

# def class
classes = ('himuro', 'ibarada', 'kanade', 'kosuke', 'yukimura')

use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")
print(device)
# load resnet model
resnet = models.resnet18(pretrained=True).to(device)
# print model
print(resnet)

import torch.nn as nn

# Change here to not relearn weights (= Transfer Learning) 
# Fine-Tuning is not fixed.
for param in resnet.parameters():
    param.requires_grad = False

resnet.fc = nn.Linear(512,5)

# print model
print(resnet)

def check_gpu():

  use_gpu = torch.cuda.is_available()
  unit = "cpu"

  if use_gpu:
    print ("-------GPUモード--------")
    unit = "cuda"
  
  else:
    print ("-------CPUモード--------")
  
  device = torch.device(unit)

  return device

def train_model(model, dataset_loader, optimizer, device):

    # 訓練モードに切り替え
    model.train()
    
    loss_summary = 0
    correct = 0

    train_num = len(dataset_loader.dataset)

    for data, target in dataset_loader:

        # 入力データをラベルを取り出します。
        # GPU処理用に変換
        input_datas, labels = data.to(device), target.to(device)
    
        # 勾配初期化
        optimizer.zero_grad()
    
        # ① 順伝播
        predicted = model(input_datas)

        # ② 誤差計算
        loss = criterion(predicted, labels)
        loss_summary += loss.item()

        # ③ 誤差逆伝播
        loss.backward()

        # ④ パラメータ更新
        optimizer.step()
        scheduler.step()

        # ⑤ 精度計算
        pred_labels = torch.max(predicted, 1)[1]
        correct += (pred_labels == labels).sum().item()

    average_loss = loss_summary / train_num
    accuracy = correct / train_num

    return model, average_loss, accuracy

def eval_model(model, dataset_loader, device):
    
  # 評価モードに切り替え
  model.eval()
  
  loss_summary = 0
  correct = 0
  test_num = len(dataset_loader.dataset)
  
  with torch.no_grad():
    for data, target in dataset_loader:
          
      # 入力データをラベルを取り出します。
      # GPU処理用に変換
      input_datas, labels = data.to(device), target.to(device)

      # ① 順伝播
      predicted = model(input_datas)

      # ② 誤差計算
      loss = criterion(predicted, labels)
      loss_summary += loss.item()

      # ③ 精度計算
      pred_labels = torch.max(predicted, 1)
      correct += (pred_labels[1] == labels).sum().item()

  average_loss = loss_summary / test_num
  accuracy = correct / test_num
  
  return average_loss, accuracy

def plot_graph(train, test, label):
  
    plt.plot(train, label='train')
    plt.plot(test, label='test')
    plt.xlabel(label)
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()

def save_model(model, acc, max_acc, epoch, path):

    if acc > max_acc:
        torch.save(model.state_dict(), path.format(epoch))
        return acc
        
    else:
        return max_acc

def training(model, epoch_num, path, optimizer):

    max_acc = 0
    max_epoch_size = epoch_num # 何回学習を行うか

    train_losses = []
    train_acc = []
    val_losses = []
    val_acc = []

    # GPUチェック
    device = check_gpu()
    model = model.to(device)

    print("Start Training!")
    for epoch in range(max_epoch_size):
        print('Epoch: {}'.format(epoch))

        # 学習
        model, loss, acc = train_model(model, train_loader, optimizer, device)
        train_losses.append(loss)
        train_acc.append(acc)
        print("Train loss: {} acc: {} lr: {}".format(loss, acc,scheduler.get_last_lr()[0]))

        # 評価
        loss, acc = eval_model(model, test_loader, device)
        val_losses.append(loss)
        val_acc.append(acc)
        print("Val loss: {} acc: {} lr: {}\n\n".format(loss, acc,scheduler.get_last_lr()[0]))

        # 精度が向上したときにモデルを保存します。
        max_acc = save_model(model, acc, max_acc, epoch, path)

    print("finish Training")
    print('Max Accuracy {}'.format(max_acc))

    # 誤差の遷移をplot
    plot_graph(train_losses, val_losses, 'Loss')

    # 精度の遷移をplot
    plot_graph(train_acc, val_acc, 'Accuracy')

"""
# Parameter settings.
　epoch: number of training sessions
　path: location where the model is stored
　criterion: loss function
　model: model
"""
epoch = 10
path = "./model/rikekoi_{:02}.pth"
criterion = torch.nn.CrossEntropyLoss()
optimizer= torch.optim.SGD(resnet.parameters(), lr=0.001, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.1)


# 学習開始!
training(resnet, epoch, path, optimizer)