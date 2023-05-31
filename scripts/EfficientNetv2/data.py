'@author:NavinKumarMNK'
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms
import cv2

"""
@Type : image_id - label => .csv file [5 classes] 
"""

class EfficientNetv2Dataset(Dataset):
    def __init__(self, batch_size:int,
                    data_path, annotation_path) -> None:
        super(EfficientNetv2Dataset, self).__init__()
        self.data_path = data_path
        self.annotation = open(annotation_path, 'r').read().splitlines()
        self.batch_size = int(batch_size)

        self.preprocessing = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(0.25),
            transforms.RandomVerticalFlip(0.25),
            transforms.RandomRotation(30, expand=True),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index:int):
        image_path, label = self.annotation[index].split(' ')
        image = cv2.imread(os.path.join(self.data_path, image_path))
        image = self.preprocessing(image)
        label = torch.tensor(int(label))
        return image, label
    
class EfficientNetv2DataModule(pl.LightningDataModule):
    def __init__(self, batch_size:int, num_workers:int,
                    data_path, annotation_path, num_classes:int) -> None:
        super(EfficientNetv2DataModule, self).__init__()
        self.annotation_path = annotation_path
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.num_classes = num_classes
        self.data_path = data_path

    def setup(self, stage=None):
        full_dataset = EfficientNetv2Dataset(self.batch_size,
                                           self.data_path, self.annotation_path)
        train_size = int(0.9 * len(full_dataset))
        val_size = int(0.1 * len(full_dataset))
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_size, val_size])
        
    def train_dataloader(self):
        class_samples = [0] * self.num_classes
        for _, label in self.train_dataset:
            class_samples[label] += 1
        total_samples = sum(class_samples)
        class_weights = [total_samples / (self.num_classes * count) for count in class_samples]

        # Create a weighted sampler based on class weights
        sampler =  WeightedRandomSampler(class_weights, len(self.train_dataset), replacement=True)

        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                           shuffle=True, drop_last=True, pin_memory=True, sampler=sampler) 

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                           shuffle=False, drop_last=True, pin_memory=True)

if __name__ == '__main__':
    dataset = EfficientNetv2DataModule(
                    batch_size=64,
                    num_workers=4,
                    num_classes=5,
                    data_path=None, 
                    annotation_path=None)
    dataset.setup()
    
    train_loader = dataset.train_dataloader()
    for i, (x, y) in enumerate(train_loader):
        print(x.shape, y.shape)
        break