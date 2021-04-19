# coding=utf-8
# Linux系统下，如出错，上面的utf-8如出错改为gbk

import matplotlib.pyplot as plt
# import torch as tf
import torch
from torch.utils.data import DataLoader

from preprocessing_image import ImageDataGenerator_test
from UNet import Unet

'''
测试函数支持多张图片读取，但是代码只支持单张查看，
需要多图片或者保存图片的话需要修改代码，改一下imshow里面的格式就行
看图窗口按‘s’可以保存图片

注意：使用交叉熵损失函数的时候程序要进行一定的更改
model = Unet(3, 1).to(device) -> model = Unet(3, 3).to(device)
predicerd_img = predicerd_img[0].squeeze(0)后面加上a = predicerd_img[0, ...] * 0.299 + predicerd_img[1, ...] * 0.587 + predicerd_img[2, ...] * 0.114
plt.imshow((predicerd_img[0] > 0.5).float().cpu().numpy()) -> plt.imshow((a > 0.5).float().cpu().numpy())
'''

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
print("Using {} device".format(device))

img_path = 'data/test'  # 测试集路径
model_path = "model.pth"  # 保存的模型文件的路径


def test_fun(img_path, model_path="model.pth"):  # 测试函数
    predicerd_img = []
    test_data = ImageDataGenerator_test(img_path)  # 读取测试集数据
    test_dataloader = DataLoader(test_data, batch_size=1)
    for batch, img in enumerate(test_dataloader):
        img = img.permute(0, 3, 1, 2).to(device) / 255
        model = Unet(3, 1).to(device)  # 装载模型
        # 装载模型参数，注意，如果测试机器和训练机器上的话GPU参数不一致的话可能会报错，所以需要用到下面的语句
        model.load_state_dict(torch.load(model_path))
        model = torch.nn.DataParallel(model, device_ids=[0])  # [0]指的是GPU的ID
        model.eval()
        with torch.no_grad():
            pred = model(img)  # 预测的结果
            predicerd_img.append(pred)
    predicerd_img = predicerd_img[0].squeeze(0)  # 这里包含了Batch维以及通道维，需要先进行降维再转换格式
    plt.imshow((predicerd_img[0] > 0.5).float().cpu().numpy())
    plt.show()


if __name__ == "__main__":
    test_fun(img_path, model_path)
