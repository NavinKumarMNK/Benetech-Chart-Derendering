'@author: NavinKumarMNK'
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
import torch.nn as nn
import lightning.pytorch as pl
import wandb
from torchvision import models
from scripts.EfficientNetv2.infernece import TensorRTInference, ToTensorRT
from torchsummary import summary

class EfficientNetv2(pl.LightningModule):
    def __init__(self, input_dim, model_path:str=None, model_name='efficientnet-v2-s', 
                 pretrained:bool=True, infer_tensorrt:bool=False, 
                 n_classes:int=1000, infer_batch_size:int=1):
        super(EfficientNetv2, self).__init__()
        self.model_path = model_path
        self.example_input_array = torch.rand(1, 3, input_dim, input_dim)
        self.n_classes = n_classes
        self.infer_tensorrt = infer_tensorrt
        self.infer_batch_size = infer_batch_size
        self.input_dim = input_dim
        self.model_family = {
            'efficientnet-v2-s': models.efficientnet_v2_s,
            'efficientnet-v2-m': models.efficientnet_v2_m,
            'efficientnet-v2-l': models.efficientnet_v2_l,
        }
        self.model_name = model_name
        self.model_type = self.model_family[self.model_name]
        
        try:
            if self.infer_tensorrt:
                print('Loading TensorRT model')
                self.example_input_array[0] = infer_batch_size
                self.model = TensorRTInference(self.model_path+'.trt',
                                               batch_size=infer_batch_size,
                                               input_dim=input_dim,)
                
            else:
                print('Loading PyTorch model')
                self.model = torch.load(model_path+'.pt')
                self.output_dim = self.model.features[-1][0].out_channels
        except FileNotFoundError as error:
            print(error)
            print('Creating new model')
            self.create_model(model_name, pretrained)
            if self.infer_tensorrt != True:
                assert n_classes is not None, 'Please provide number of classes for the model'
                self._make_classifier(limit=64)
                self.save_model()
                
        ## set model to train
        self.model.train()

    def _make_classifier(self, limit:int=64):
        self.model.classifier = nn.Sequential(
            nn.Linear(self.output_dim, 
                      self.output_dim*2, bias=False),
            nn.BatchNorm1d(self.output_dim*2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(self.output_dim*2, 
                      self.output_dim, bias=False),
            nn.BatchNorm1d(self.output_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            )

        temp = self.output_dim
        while temp > limit:
            self.model.classifier.extend([
                nn.Linear(temp,
                          temp//4 if temp//4 > limit else self.n_classes),
                nn.ReLU(),
                nn.Dropout(0.4),
            ])
            temp = temp//4

        for layer in self.model.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight)
                try:
                    nn.init.zeros_(layer.bias)
                except:
                    pass
        
    def forward(self, x:torch.Tensor):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss)
        self.log('val_acc', acc)

    def on_validation_epoch_end(self) -> None:
        ## 

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def to_tensorrt(self, batch_size:int=1, input_dim:int=256):
        ToTensorRT(self.model_path, )

    def create_model(self, model_name:str, pretrained:bool=True):
        if pretrained:
            self.weights = {
                'efficientnet-v2-s': models.EfficientNet_V2_S_Weights,
                'efficientnet-v2-m': models.EfficientNet_V2_M_Weights,
                'efficientnet-v2-l': models.EfficientNet_V2_L_Weights,
            }
            self.weights_type = self.weights[model_name]
        else :
            self.weights_type = None
        self.model = self.model_type(weights=self.weights_type)
        self.output_dim = self.model.features[-1][0].out_channels
            


    def save_model(self):
        torch.save(self.model, self.model_path+'.pt')
    
    def finalize(self):
        self.save_model()
        self.to_torchscript(self.model_path+'_script.pt', method='script', example_inputs=self.example_input_array)
        self.to_onnx(self.model_path+'.onnx', self.example_input_array, export_params=True)
        self.to_tensorrt()

if __name__ == '__main__':
    model = EfficientNetv2(model_name='efficientnet-v2-s',
                           model_path='/workspace/Benetech-Kaggle-Competition/models/efficientnet-v2-s',
                           input_dim=256,
                           n_classes=5,
                           infer_tensorrt=True,
                           infer_batch_size=1).cuda()
    
    output = model(torch.rand(1, 3, 256, 256).numpy())
    print(output)
    
    #summary(model, (3, 256, 256))
    #model.finalize()
