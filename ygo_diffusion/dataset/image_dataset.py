# coding=utf-8

import torchvision
from PIL import Image
from torch.utils.data import Dataset
import os


class ImageDataset(Dataset):
    def __init__(self, folder_path, resize=None):
        self.folder_path = folder_path
        self.image_names = os.listdir(folder_path)
        self.resize = resize

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        # 只处理图片文件
        if not image_name.endswith('.jpg') and not image_name.endswith('.png'):
            return None
        image = Image.open(os.path.join(self.folder_path, image_name))
        if self.resize is not None:
            image = self.resize(image)
        image = torchvision.transforms.ToTensor()(image)
        return image
