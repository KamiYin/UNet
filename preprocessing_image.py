import os

import numpy as np
# import torch as tf
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms

'''
Coding ————Kami
'''


class ImageDataGenerator(data.Dataset):
    def __init__(self, file_path=[], img_size=[256, 256], times=1):
        if len(file_path) != 2:
            raise ValueError("路径格式：[图片, 标签], 图片格式：[高，宽]，数据扩增倍数（默认不扩增）")
        self.imgs = file_path[0]  # 此处保存的是文件夹路径
        self.labels = file_path[1]
        self.img_H = img_size[0]
        self.img_W = img_size[1]
        self.times = times  # 扩增倍数
        self.img_path = os.listdir(self.imgs)  # 这里返回文件夹下所有文件的文件名
        self.label_path = os.listdir(self.labels)

    def __getitem__(self, index):
        if self.times == 1:
            img = torch.from_numpy(
                np.array(Image.open(self.imgs + '/' + self.img_path[index]).resize((self.img_H, self.img_W)).convert('RGB')))  # 默认所有输入都是三通道，输出都是单通道
            label = torch.from_numpy(
                np.array(Image.open(self.labels + '/' + self.label_path[index]).resize((self.img_H, self.img_W)).convert('L'))).unsqueeze(0)
            return [img, label]
        else:
            img = Image.open(
                self.imgs + '/' + self.img_path[int(index/self.times)]).resize((self.img_H, self.img_W)).convert('RGB')  # 索引/倍数，保证每张图都能被读取到
            label = Image.open(
                self.labels + '/' + self.label_path[int(index/self.times)]).resize((self.img_H, self.img_W)).convert('L')
            angle = transforms.RandomRotation.get_params(
                [-180, 180])  # 旋转随机角度，范围为[-180, 180]
            img = torch.from_numpy(np.array(img.rotate(angle)))
            label = torch.from_numpy(
                np.array(label.rotate(angle))).unsqueeze(0)
            return [img, label]

    def __len__(self):
        return len(self.img_path)*self.times  # 长度乘倍数，保证索引范围不会出错


class ImageDataGenerator_test(data.Dataset):  # 青春版ImageDataGenerator
    def __init__(self, file_path, img_size=[256, 256]):
        self.imgs = file_path
        self.img_H = img_size[0]
        self.img_W = img_size[1]
        self.img_path = os.listdir(self.imgs)

    def __getitem__(self, index):
        img = torch.from_numpy(
            np.array(Image.open(self.imgs + '/' + self.img_path[index]).resize((self.img_H, self.img_W)).convert('RGB')))
        return img

    def __len__(self):
        return len(self.img_path)
