import torch
import cv2
import os
from torch.utils.data import Dataset
import random
import numpy as np
from numpy.random import RandomState
import settings

class ISBI_Dataset(Dataset):
    def __init__(self, name):
        # 初始化函数，读取所有data_path下的图片
        self.rand_state = RandomState(66)# 设置随机种子
        self.root_dir = os.path.join(settings.data_dir, name)# 数据集目录
        self.mat_files = os.listdir(os.path.join(self.root_dir,"image"))# 数据名list
        self.file_num = len(self.mat_files)# 数据集大小

    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip
        
    def __getitem__(self, index):
        # 根据index读取图片
        file_name = self.mat_files[index % self.file_num]
        # 根据image_path生成label_path
        img_path = os.path.join(self.root_dir,"image", file_name)

        label_path = os.path.join(self.root_dir,"label", file_name)
        # 读取训练图片和标签图片，读灰度图,归一化
        image = cv2.imread(img_path,0).astype(np.float32) / 255
        label = cv2.imread(label_path,0).astype(np.float32) / 255
        image = np.reshape(image, (1,image.shape[0],image.shape[1]))
        label = np.reshape(label, (1,label.shape[0],label.shape[1]))
        # print(image.shape)
        # 通道转换
        # image = np.transpose(image,(2,0,1))
        # label = np.transpose(label,(2,0,1))
        # 随机进行数据增强，为2时不做处理
        flipCode = random.choice([-1, 0, 1, 2])
        if flipCode != 2:
            image = self.augment(image, flipCode)
            label = self.augment(label, flipCode)
        sample = {'image': image, 'label': label}
        return sample

    def __len__(self):
        # 返回训练集大小
        return self.file_num


class TestDataset(Dataset):
    def __init__(self, name):
        # 初始化函数，读取所有data_path下的图片
        self.rand_state = RandomState(66)
        self.root_dir = os.path.join(settings.data_dir, name)
        self.mat_files = os.listdir(os.path.join(self.root_dir,"image"))
        self.file_num = len(self.mat_files)

    def __getitem__(self, index):
        file_name = self.mat_files[index % self.file_num]
        # 根据image_path生成label_path
        img_path = os.path.join(self.root_dir,"image", file_name)
        # 读取训练图片和标签图片,归一化,灰度读哈
        image = cv2.imread(img_path,0).astype(np.float32) / 255
        image = np.reshape(image, (1, image.shape[0], image.shape[1]))
        # 通道转换
        # image = np.transpose(image,(2,0,1))
        sample = {'image': image, 'idx': file_name}
        return sample # 返回照片的名称，方便后期保存

    def __len__(self):
        # 返回训练集大小
        return self.file_num

    
if __name__ == "__main__":
    isbi_dataset = ISBI_Dataset("train")
    print("数据个数：", len(isbi_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=2, 
                                               shuffle=True)
    for image, label in train_loader:
        print(image.shape)