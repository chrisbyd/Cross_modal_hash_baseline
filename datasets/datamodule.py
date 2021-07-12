from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torch.utils.data as torchdata
import os
from  datasets import load_data

mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)

class HashDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        self.test_transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        root = config["data_root"]


    def setup(self, stage):

        self.t_data_set, self.q_data_set, self.g_data_set = load_data(name = self.config['dataset_name'])


    def train_dataloader(self):
        train_loader = DataLoader(self.t_data_set, batch_size= self.config['batch_size'], shuffle= True, num_workers =4)
        return train_loader 

    def query_loader(self):
        query_loader = DataLoader(self.q_data_set, batch_size=self.config['test_batch_size'], shuffle=False, num_workers=4)
        return query_loader

    def gallery_loader(self):
        gallery_loader = DataLoader(self.g_data_set, batch_size=self.config['test_batch_size'], shuffle=False, num_workers=4)
        return gallery_loader

    def test_dataloader(self):
        test_loader = DataLoader(self.test_set, batch_size = self.config['batch_size'], shuffle = False, num_workers = 4)
        return test_loader



