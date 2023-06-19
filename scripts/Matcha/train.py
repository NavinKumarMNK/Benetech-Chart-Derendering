'@author: NavinKumarMNK'
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
import torch.nn as nn
import lightning.pytorch as pl
import wandb
from torchvision import models
from torchsummary import summary

from transformers import PreTrainedTokenizerBase, PreTrainedModel
from transformers import Pix2StructConfig, Pix2StructForConditionalGeneration, AutoProcessor
from scripts.utils.config import Config, NEW_TOKENS

from typing import Dict
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning import callbacks as pl_callbacks

from scripts.Matcha.data import BeneTechDataModule
from scripts.Matcha.model import MatChaFineTuned

import json
import pandas as pd
from typing import Dict, List, Tuple

if __name__ == '__main__':
    with open("/kaggle/input/benetech-metadata/train_data(2).json", "r") as fl:
        train_dataset = json.load(fl)
        
    with open("/kaggle/input/benetech-metadata/val_data(1).json", "r") as fl:
        val_dataset = json.load(fl)

    
    model = Pix2StructForConditionalGeneration.from_pretrained("google/matcha-base")
    processor = AutoProcessor.from_pretrained("google/matcha-base", is_vqa=False)

    processor.image_processor.size = {
        "height": Config.image_size,
        "width": Config.image_size,
    }

    example_str = train_dataset[0]['ground_truth']
    temp_ids = processor.tokenizer(example_str).input_ids
    print("ids:", temp_ids)
    print("tokenized:", processor.tokenizer.tokenize(example_str))
    print("decoded:", processor.tokenizer.decode(temp_ids))
    num_added = processor.tokenizer.add_tokens(["\n"] + NEW_TOKENS)
    model.resize_token_embeddings(len(processor.tokenizer))
    print(len(processor.tokenizer))
    print(num_added, "tokens added")

    processor.eos_token_id = processor.tokenizer.convert_tokens_to_ids(["<Y_END>"])[0]
    processor.bos_token_id  = processor.tokenizer.convert_tokens_to_ids(["<|BOS|>"])[0]
    model.config.eos_token_id = processor.eos_token_id
    model.config.bos_token_id = processor.bos_token_id

    dataset = BeneTechDataModule(train_dataset, val_dataset, processor, batch_size=2, num_workers=2)
    dataset.setup()

    model_train = MatChaFineTuned(
        processor=processor,
        model=model,
        model_path="/kaggle/working/matcha",
    )

    callbacks = [
        pl_callbacks.ModelCheckpoint(
            monitor='val_loss',
            dirpath='/kaggle/working/checkpoints',
            filename='matcha-{epoch:02d}-{val_loss:.2f}',
            save_top_k=1,
            mode='min',
        ),
        pl_callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            mode='min',
        ),
        pl_callbacks.LearningRateMonitor(logging_interval='epoch'),
        pl_callbacks.TQDMProgressBar(),
    ]

    from pytorch_lightning.loggers import WandbLogger
    from pytorch_lightning import callbacks as pl_callbacks

    wandb_logger = WandbLogger(project="matcha", name="matcha")
    wandb_logger.watch(model, log='all', log_freq=100)

    trainer = pl.Trainer(
        accelerator="cuda",
        max_epochs=10,
        devices=2,
        callbacks=callbacks,
        logger=wandb_logger,
        log_every_n_steps=1,
        limit_train_batches=0.1,
        accumulate_grad_batches=8,
        num_sanity_val_steps=10,
        precision=16,
    )

    trainer.fit(model_train, train_dataloaders=dataset.train_dataloader(),
                val_dataloaders=dataset.val_dataloader())

    model_train.model.save_pretrained("/kaggle/working/deplot.pt")
    model_train.processor.save_pretrained("/kaggle/working/deplot.pt")