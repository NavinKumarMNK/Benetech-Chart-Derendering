'@author:NavinKumarMNK'
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset as TorchDataset, DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms
import cv2
from scripts.utils.config import Config
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd

graph_types_rev = {
    "dot":"d",
    "horizontal_bar":"h",
    "scatter":"s",
    "line":"l",
    "vertical_bar":"v"
}

def augments():
        return A.Compose([
            A.Resize(width=Config.image_height, height=Config.image_width),
            A.Normalize(
                mean=[0, 0, 0], 
                std=[1, 1, 1],
                max_pixel_value=255,
            ),
            ToTensorV2(),
        ])


class BeneTechDataset(TorchDataset):
    def __init__(self, dataset, processor, ):
        self.dataset = dataset
        self.processor = processor
        self.augmenter = augments()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx] 
        if item['source'] == 'extra':
            path = "/kaggle/input/benetech-extra-generated-data/"
            path = path + "graphs_" + graph_types_rev[item['chart-type']] + "/"
            path = path + item['path'].split('-')[-1]
        else:
            path = "/kaggle/input/benetech-making-graphs-accessible/train/images/"
            path = path + item['path']

        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = self.augmenter(image=image)['image']
        encoding = self.processor(
            images=torch.Tensor(image),
            return_tensors="pt", 
            add_special_tokens=True, 
            max_patches=Config.max_patches
        )

        encoding = {k:v.squeeze() for k,v in encoding.items()}
        encoding["text"] = item['ground_truth']
        
        return encoding


class BeneTechDataModule(pl.LightningDataModule):
    def __init__(self, dataset, val_dataset, processor, batch_size=1, num_workers=1):
        super().__init__()
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.processor = processor
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = BeneTechDataset(self.dataset, self.processor)
        self.val_dataset = BeneTechDataset(self.dataset, self.processor)
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.collate_fn
        )

    def collate_fn(self, batch):
        print("collate_fn")
        print(batch)
        new_batch = {"flattened_patches": [], "attention_mask": []}
        texts = [item["text"] for item in batch]
        encoding = self.processor(
            text=texts,
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            add_special_tokens=True,
            max_length=Config.max_length
        )
        new_batch["labels"] = encoding.input_ids
        for item in batch:
            new_batch["flattened_patches"].append(item["flattened_patches"])
            new_batch["attention_mask"].append(item["attention_mask"])
        new_batch["flattened_patches"] = torch.stack(new_batch["flattened_patches"])
        new_batch["attention_mask"] = torch.stack(new_batch["attention_mask"])
        return new_batch

if __name__ == '__main__':
    data = BeneTechDataModule(train_dataset, val_dataset, processor, batch_size=2, num_workers=1)
    data.setup()
    for sample in data.val_dataloader():
        print(sample)
        break