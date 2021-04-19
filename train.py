# coding=utf-8
# Linux系统下，如出错，上面的utf-8改为gbk

# import torch as tf
import torch
from torch.nn import CrossEntropyLoss

from loss import DiceLoss, FocalLoss
from preprocessing_image import ImageDataGenerator
from UNet import Unet

'''
注意：使用交叉熵损失函数的时候程序要进行一定的更改
loss_fn = CrossEntropyLoss()
model = Unet(3, 1).to(device) -> model = Unet(3, 3).to(device)
loss_fn(pred, label) -> loss_fn(pred, label.squeeze(0).long()) //两处
'''
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
print("Using {} device".format(device))

img_path = 'data/train'  # 训练集路径
label_path = 'data/label'  # 标签路径
val_path = 'data/val'  # 验证集路径
val_label_path = 'data/val_label'  # 验证集标签路径
batch_size = 1
epoch = 1
img_H = 256  # 图像的高
img_W = 256  # 图像的宽
times = 10  # 图像扩增倍数
loss_fn = FocalLoss()  # 损失函数选择：DiceLoss，FocalLoss，CrossEntropyLoss


def train_fun(img_path, label_path, val_path, val_label_path,  # 训练函数
              batch_size=1, epoch=5,
              img_H=256, img_W=256, times=5,
              loss_fn=DiceLoss()):
    training_data = ImageDataGenerator(  # 读取数据，路径格式：[图片, 标签], 图片格式：[高，宽]，数据扩增倍数（默认不扩增）
        [img_path, label_path], [img_H, img_W], times=times)
    val_data = ImageDataGenerator([val_path, val_label_path])
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    val_dataloader = DataLoader(val_data, batch_size=1)

    model = Unet(3, 1).to(device)  # 装载模型,输入为三通道，输出为单通道

    optimizer = torch.optim.Adam(model.parameters())  # 定义优化器
    train_loss, val_loss = [], []  # 记录全程的Loss变化
    batch_num = 0
    for i in range(epoch):
        for batch, (img, label) in enumerate(train_dataloader):  # 以Batch为单位训练网络
            img, label = img.permute(0, 3, 1, 2).to(  # 由于Image和Tensor储存图像的区别，需要进行维度转换至[B, C, H, W]
                device) / 255, label.to(device) / 255
            model.train()  # 训练用.train()
            pred = model(img)  # 预测图像
            loss = loss_fn(pred, label)  # 计算loss
            optimizer.zero_grad()
            loss.backward()  # 梯度回传
            optimizer.step()
            train_loss.append(loss.item())  # 保存梯度，.item()防止爆显存
            size = len(train_dataloader.dataset)
            if batch % 1 == 0:
                loss, current = loss.item(), batch * len(img) + 1
                print(
                    f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]  epoch:{i+1}/ {epoch}")

            size = len(val_dataloader.dataset)
            model.eval()  # 测试用.eval()
            correct = 0
            with torch.no_grad():  # 验证集，记录网络在验证集上的loss
                for batch, (img, label) in enumerate(val_dataloader):
                    img, label = img.permute(0, 3, 1, 2).to(
                        device) / 255, label.to(device) / 255
                    pred = model(img)
                    correct += loss_fn(pred, label).item()
            val_loss.append(correct / size)
            batch_num = batch_num + 1
    torch.save(model.state_dict(), "model.pth")  # 保存模型
    print("Saved PyTorch Model State to model.pth")

    x = range(batch_num)  # 显示Loss变化
    y1 = train_loss
    y2 = val_loss
    plt.plot(x, y1, '-')
    plt.plot(x, y2, '-')
    plt.show()


if __name__ == "__main__":
    train_fun(img_path, label_path, val_path, val_label_path,
              batch_size, epoch, img_H, img_W, times, loss_fn)
