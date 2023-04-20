# coding=utf-8

import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
from ygo_diffusion.dataset import ImageDataset
from ygo_diffusion.net import *

def print_hi(name):
    # 在下面的代码行中使用断点来调试脚本。
    print(f'Hi, {name}')  # 按 ⌘F8 切换断点。

def test():
    image = Image.open("/Users/cyendra/Desktop/workshop/ygodream/ygo_diffusion/images/cards/10321588.png")
    # 显示图片
    plt.imshow(image)

    # x_start = transform(image).unsqueeze(0)
    # plot(image, [get_noisy_image(x_start, torch.tensor([t])) for t in [0, 50, 100, 150, 199]])

    resize = torchvision.transforms.Resize((61, 42))
    dataset = ImageDataset('images/cards', resize)

    images = [dataset[i] for i in range(16)]
    grid = torchvision.utils.make_grid(images, nrow=4, padding=10, pad_value=1)  # 指定行数为4，边距为10，填充颜色为白色

    plt.figure(figsize=(10, 10))  # 指定图像大小为10x10英寸
    plt.imshow(grid.permute(1, 2, 0))  # 调整Tensor的维度顺序，以适应imshow函数的输入格式
    plt.axis("off")  # 关闭坐标轴
    plt.show()  # 显示图像


def transforms(examples):
    examples["pixel_values"] = [transform(image.convert("L")) for image in examples["image"]]
    del examples["image"]

    return examples


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    # load dataset from the hub
    resize = torchvision.transforms.Resize((61, 42))
    dataset = ImageDataset('images/cards', resize)
    # image_size = 28
    # channels = 3
    batch_size = 128




    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    batch = next(iter(dataloader))
    print(dataset[0])
