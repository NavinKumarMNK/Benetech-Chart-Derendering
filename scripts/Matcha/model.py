'@author: NavinKumarMNK'
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
import lightning.pytorch as pl

from transformers import PreTrainedTokenizerBase, PreTrainedModel
from transformers import Pix2StructConfig, Pix2StructForConditionalGeneration, AutoProcessor
from scripts.utils.config import Config

from typing import Dict
class MatChaFineTuned(pl.LightningModule):
    def __init__(self, processor,
                    model,
                    model_path:str=None,):
        super(MatChaFineTuned, self).__init__()
        
        self.processor = processor
        self.model = model
        self.model_path = model_path  
        self.val_loss = []
        
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx:int) -> torch.Tensor:
        labels = batch['labels']
        flattened_patches = batch['flattened_patches']
        attention_mask = batch['attention_mask']
        outputs = self.model(
            flattened_patches=flattened_patches, 
            attention_mask=attention_mask, 
            labels=labels)
        loss = outputs.loss
        self.log('train_loss', loss)
        return loss
            
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx:int) -> None:
        labels = batch['labels']
        flattened_patches = batch['flattened_patches']
        attention_mask = batch['attention_mask']
        outputs = self.model(
                flattened_patches=flattened_patches, 
                attention_mask=attention_mask, 
                labels=labels)
        loss = outputs.loss
        self.val_loss.append(loss)
        self.log('val_loss', loss)
        return {"val_loss":loss}
    
    def on_validation_epoch_end(self)-> None:
        avg_loss = torch.stack([x for x in self.val_loss]).mean()
        self.log('val/loss_epoch', avg_loss)
        self.log('val/loss', avg_loss, prog_bar=True)
        self.val_loss = []
    
            
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=0, last_epoch=-1)
        return [optimizer], [scheduler]

    def save_model(self):
        torch.save(self.model, self.model_path+'.pt')