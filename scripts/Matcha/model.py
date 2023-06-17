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
    def __init__(self, processor:PreTrainedTokenizerBase,
                    model:PreTrainedModel,
                    input_dim:int=512,
                    model_path:str=None,):
        super(MatChaFineTuned, self).__init__()
        
        self.processor = processor
        self.model = model
        self.input_dim = input_dim
        self.model_path = model_path

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx:int) -> torch.Tensor:
        labels = batch['labels']
        flattened_patches = batch['flattened_patches']
        attention_mask = batch['attention_mask']
        outputs = self.model(flattened_patches, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        self.log('train_loss', loss)
        return loss

    def validataion_step(self, batch: Dict[str, torch.Tensor], batch_idx:int,
                         dataset_indx: int=0) -> None:
        labels = batch['labels']
        flattened_patches = batch['flattened_patches']
        attention_mask = batch['attention_mask']
        outputs = self.model(flattened_patches, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-5, weight_decay=0.001)
        return optimizer      

    def save_model(self):
        torch.save(self.model, self.model_path+'.pt')
        
if __name__ == '__main__':
    config = Pix2StructConfig.from_pretrained("google/matcha-base")
    config.max_length = Config.max_length

    model = Pix2StructForConditionalGeneration.from_pretrained("google/matcha-base")
    processor = AutoProcessor.from_pretrained("google/matcha-base", is_vqa=False)

    model = MatChaFineTuned(
        processor=processor,
        model=model,
        gt_df=None,
        input_dim=512,
        model_path="models/matcha",
    ).cuda()
