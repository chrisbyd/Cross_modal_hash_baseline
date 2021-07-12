import numpy as np
from PIL import Image
import torch.utils.data as torchdata
import os



def default_loader(path):
    return Image.open(path).convert('RGB')


class HashDataset(torchdata.Dataset):
    def __init__(self, image_dir,images,texts, labels, transform=None, loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.images = images
        self.texts = texts
        self.labels = labels
        self.img_dir = image_dir


    def __getitem__(self, index):

        img_path, label, text = self.images[index], self.labels[index], self.texts[index]
        img_path = img_path.strip()
        img_path = os.path.join(self.img_dir, img_path)
        # img
        img = self.loader(img_path)
        if self.transform is not None:
            img = self.transform(img)
        # text

        return img, text, label

    def __len__(self):
        return len(self.images)