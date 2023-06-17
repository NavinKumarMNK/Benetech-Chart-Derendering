'@author:NavinKumarMNK'
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms
import cv2
from scripts.utils.config import Config
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd

class BeneTechDataset(Dataset):
    def __init__(self, dataset, processor, augments=None):
        self.dataset = dataset
        self.processor = processor

    def augments():
        return A.Compose([
            A.Resize(width=Config['IMG_SIZE'][0], height=Config['IMG_SIZE'][1]),
            A.Normalize(
                mean=[0, 0, 0],
                std=[1, 1, 1],
                max_pixel_value=255,
            ),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = cv2.imread(item['image'])
        if self.augments:
            image = self.augments(image=image)['image']
        
        encoding = self.processor(
            images=image,
            return_tensors="pt", 
            add_special_tokens=True, 
            max_patches=Config['MAX_PATCHES']
        )
        
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        encoding["text"] = item["label"]
        return encoding

class BeneTechDataModule(pl.LightningDataModule):
    def __init__(self, dataset, processor, batch_size=1, num_workers=1):
        super().__init__()
        self.dataset = dataset
        self.processor = processor
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.dataset = pd.read_csv(self.dataset)
        self.dataset = BeneTechDataset(self.dataset, self.processor)
        train_len = int(0.8*len(self.dataset))
        val_len = len(self.dataset) - train_len
        self.train_dataset, self.val_dataset = random_split(self.dataset, [train_len, val_len])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=False
        )

    def collate_fn(self, batch):
        new_batch = {"flattened_patches": [], "attention_mask": []}
        texts = [item["text"] for item in batch]
        encoding = self.processor(
            texts,
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            add_special_tokens=True,
            max_length=Config['MAX_LENGTH']
        )
        new_batch["labels"] = encoding.input_ids
        for item in batch:
            new_batch["flattened_patches"].append(item["pixel_values"])
            new_batch["attention_mask"].append(item["attention_mask"])
        new_batch["flattened_patches"] = torch.stack(new_batch["flattened_patches"])
        new_batch["attention_mask"] = torch.stack(new_batch["attention_mask"])
        return new_batch