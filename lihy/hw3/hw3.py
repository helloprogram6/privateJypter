import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import cv2
import os

def readfile(path,label):
    img_dir = sorted(os.listdir(path))
    x = np.zeros((len(img_dir),128,128,3),dtype = np.uint8) # 图片存在这个维度
    y = np.zeros(len(img_dir),dtype = np.uint8) # 标签
    for i , file in enumerate(img_dir):
        x [i,:,:] = cv2.resize(cv2.imread(os.path.join(path,file)),(128,128))
        if label:
            y[i] = int(file.split('_')[0])
    if label:
        return x, y
    else:
        return y

workspace_dir ='food-11/'
print("Reading data")
train_x , train_y = readfile(os.path.join(workspace_dir,"training"),True)
print("Size of training data = {}".format(len(train_x)))
val_x , val_y = readfile(os.path.join(workspace_dir,"validation"),True)
print("Size of validation data = {}".format(len(val_x)))
test_x = readfile(os.path.join(workspace_dir,'testing'),False)
print("Size of Testing data = {}".format(len(test_x)))

train_transform = transforms.Compose([
    transforms.ToPILImage(),#
    transforms.RandomHorizontalFlip(), #水平翻转
    transforms.RandomRotation(15), # 随机旋转
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

class ImgDataset(Dataset):
    def __init__(self,x,y=None,transform=None):
        self.x = x
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self,index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y != None:
            Y = self.y[index]
            return X,Y
        else:
            return X


batch_size = 128
train_set = ImgDataset(train_x,train_y,train_transform) # 实例化train_set
val_set = ImgDataset(val_x,val_y,test_transform)
train_loader = DataLoader(train_set,batch_size = batch_size,shuffle = True)
val_loader = DataLoader(val_set,batch_size = batch_size,shuffle=False)


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [3,128,128]-->[64,128,128] 64个特征过滤器，过滤器大小为3*3*3
            nn.BatchNorm2d(64),  # 64个特征。数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [64,64,64]

            nn.Conv2d(64, 128, 3, 1, 1),  # [64,64,64]-->[128,64,64] 128个特征过滤器，过滤器大小为64*3*3
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [128,32,32]

            nn.Conv2d(128, 256, 3, 1, 1),  # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [256, 16, 16]

            nn.Conv2d(256, 512, 3, 1, 1),  # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [512, 8, 8]

            nn.Conv2d(512, 512, 3, 1, 1),  # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [512, 4, 4]
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 11)
        )

    def forward(self, x):
        out = self.cnn(x)#[128,3,128,128]-->[128,512,4,4]
        out = out.view(out.size()[0], -1) #全连接层
        return self.fc(out)


model = Classifier()
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epoch = 30
for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0
    model.train()
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        train_pred = model(data[0])
        batch_loss = loss(train_pred, data[1])
        batch_loss.backward()
        optimizer.step()
        # 最大值的索引：
        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            val_pred = model(data[0])  # 验证集的pred
            batch_loss = loss(val_pred, data[1])
            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            val_loss += batch_loss.item()
            print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
                  (epoch + 1, num_epoch, time.time() - epoch_start_time, \
                   train_acc / train_set.__len__(), train_loss / train_set.__len__(), val_acc / val_set.__len__(),
                   val_loss / val_set.__len__()))

test_set = ImgDataset(test_x,transform = test_transform)
test_loader = DataLoader(test_set , batch_size = batch_size ,shuffle=False)

model_best.eval()
predicition = []
with torch.no_grad():
    for i,data in enumerate(test_loader):
        test_pred = model_best(data)
        test_label = np.argmax(test_pred.cpu().data.numpy(),axis = 1)
        for y in test_label:
            predicition.append(y)