'@author: NavinKumarMNK'
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
import torch.nn as nn
import lightning.pytorch as pl
import lightning.pytorch.loggers as pl_loggers
import lightning.pytorch.callbacks as pl_callbacks
import wandb
from torchvision import models
from scripts.EfficientNetv2.infernece import TensorRTInference, ToTensorRT
from torchsummary import summary
from scripts.EfficientNetv2.model import EfficientNetv2
from scripts.EfficientNetv2.data import EfficientNetv2DataModule


if __name__ == '__main__':
    model = EfficientNetv2(model_name='efficientnet-v2-s',
                           model_path='/workspace/Benetech-Kaggle-Competition/models/efficientnet-v2-s',
                           input_dim=256,
                           n_classes=5,
                           infer_tensorrt=True)
    
    dataset = EfficientNetv2DataModule(
                    batch_size=64,
                    num_workers=4,
                    num_classes=5,
                    data_path=None, 
                    annotation_path=None)
    dataset.setup()

    wandb_logger = pl_loggers.WandbLogger(project='Benetech-Kaggle-Competition',
                                            log_model=True,
                                            save_dir='/workspace/Benetech-Kaggle-Competition/logs',
                                            name='efficientnet-v2-s')

    callbacks = [
        pl_callbacks.ModelCheckpoint(
            monitor='val_loss',
            dirpath='/workspace/Benetech-Kaggle-Competition/models/checkpoints',
            filename='efficientnet-v2-s-{epoch:02d}-{val_loss:.2f}',
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
        accelerator='ddp',
        precision=1,
        callbacks=[pl.callbacks.ProgressBar()],
        logger=wandb_logger,
        profiler='simple',
        log_every_n_steps=1,
        flush_logs_every_n_steps=10,
        weights_summary='full',
    )


    trainer.fit(model, dataset)
    model.finalize()



