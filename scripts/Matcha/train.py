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
from scripts.utils.config import Config

from typing import Dict
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning import callbacks as pl_callbacks

from scripts.Matcha.data import BeneTechDataModule
from scripts.Matcha.model import MatChaFineTuned

import pandas as pd
from typing import Dict, List, Tuple

if __name__ == '__main__':
    config = Pix2StructConfig.from_pretrained("google/matcha-base")
    config.max_length = Config.max_length

    model = Pix2StructForConditionalGeneration.from_pretrained("google/matcha-base")
    processor = AutoProcessor.from_pretrained("google/matcha-base", is_vqa=False)

    wandb_logger = WandbLogger(project="matcha", name="matcha")
    wandb_logger.watch(model, log='all', log_freq=100)

    model = MatChaFineTuned(
        processor=processor,
        model=model,
        gt_df=None,
        input_dim=512,
        model_path="models/matcha",
    )

    callbacks = [
        pl_callbacks.ModelCheckpoint(
            monitor='val_loss',
            dirpath='/workspace/Benetech-Kaggle-Competition/models/checkpoints',
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

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=10,
        progress_bar_refresh_rate=20,
        precision=16,
        callbacks=callbacks,
        logger=wandb_logger,
        profiler='simple',
        log_every_n_steps=1,
        flush_logs_every_n_steps=10,
        weights_summary='full',
        accumulate_grad_batches=8
    )

   