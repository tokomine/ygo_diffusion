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

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((28, 28)),
        torchvision.transforms.ToTensor(),  # turn into Numpy array of shape HWC, divide by 255
        # torchvision.transforms.Lambda(lambda t: (t * 2) - 1),
    ])
    dataset = ImageDataset('images/cards', transform)

    images = [dataset[i] for i in range(16)]
    grid = torchvision.utils.make_grid(images, nrow=4, padding=10, pad_value=1)  # 指定行数为4，边距为10，填充颜色为白色

    plt.figure(figsize=(10, 10))  # 指定图像大小为10x10英寸
    plt.imshow(grid.permute(1, 2, 0))  # 调整Tensor的维度顺序，以适应imshow函数的输入格式
    plt.axis("off")  # 关闭坐标轴
    plt.show()  # 显示图像
    exit()


def transforms(examples):
    examples["pixel_values"] = [transform(image.convert("L")) for image in examples["image"]]
    del examples["image"]

    return examples


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    # test()
    # load dataset from the hub
    image_size = 28
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_size),
        torchvision.transforms.ToTensor(),  # turn into Numpy array of shape HWC, divide by 255
        torchvision.transforms.Lambda(lambda t: (t * 2) - 1),
    ])

    dataset = ImageDataset('images/cards', transform)
    batch_size = 128
    channels = 3

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    results_folder = Path("data/results")
    results_folder.mkdir(exist_ok=True)
    save_and_sample_every = 1

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Unet(
        dim=image_size,
        channels=channels,
        dim_mults=(1, 2, 4,)
    )
    model.to(device)

    optimizer = Adam(model.parameters(), lr=1e-3)

    epochs = 5

    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()

            batch_size = batch.shape[0]
            batch = batch.to(device)

            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()

            loss = p_losses(model, batch, t, loss_type="huber")

            print("Step {} Loss:{}".format(step, loss.item()))

            loss.backward()
            optimizer.step()

            # save generated images
            if step != 0 and step % save_and_sample_every == 0:
                milestone = step // save_and_sample_every
                batches = num_to_groups(4, batch_size)
                all_images_list = list(
                    map(lambda n: sample(model, image_size=image_size, batch_size=n, channels=channels), batches))
                all_images = torch.cat(all_images_list, dim=0)
                all_images = (all_images + 1) * 0.5
                save_image(all_images, str(results_folder / f'sample-{milestone}.png'), nrow=6)
                print("Model save to {}".format(str(results_folder / f'sample-{milestone}.png')))
