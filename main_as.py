import nibabel as nib
import numpy as np
import pandas as pd


import wandb
import os
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from medcam import medcam
from scipy.interpolate import interpn
import shap 
import numpy as np
from conv3D.model import AdniModel
# from dataset import odule
from unimodal_dataset import ASDataModule

from ResNet.model import ResNetModel

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = device.type
from torch import nn
from torchvision.models import resnet18 

class ResNetModel(nn.Module):
    def __init__(self, in_channels=8):  # Update in_channels to match the input data
        super(ResNetModel, self).__init__()
        self.resnet = resnet18(pretrained=False)
        # Replace the first convolutional layer to accept the input channels
        self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Other operations...

    def forward(self, x):
        # Example forward pass
        x = self.resnet(x)
        # Other operations...
        return x
    
def main_conv3d(wandb, wandb_logger):
    '''
    main function to run the conv3d architecture
    '''
    seed_everything(23)
    # get the model
    model = AdniModel(in_channels=8)

    # load the data
    data = ASDataModule()

    # Optional
    wandb.watch(model, log="all")

    # train the network
    trainer = Trainer(max_epochs=15, logger=wandb_logger, log_every_n_steps=1, accelerator=device, devices=1)
    trainer.fit(model, data)


def main_resnet(wandb, wandb_logger):
    '''
    main function to run the resnet architecture
    '''
    seed_everything(23)
    # ge the model
    model = ResNetModel(in_channels=8)

    # load the data
    data = ASDataModule()

    # Optional
    wandb.watch(model, log="all")

    # train the network
    trainer = Trainer(max_epochs=15, logger=wandb_logger, log_every_n_steps=1, accelerator=device, devices=1)
    trainer.fit(model, data)



if __name__ == '__main__':

    # create wandb objects to track runs
    # wandb.init(project="ncanda-imaging")
    # wandb.config = {
    #     "learning_rate": 1e-4,
    #     "epochs": 5,
    #     "batch_size": 1
    # }

    wandb_logger = WandbLogger(wandb.init(project="ncanda-emily", entity="ewesel"))

    # run conv3d
    # main_conv3d(wandb, wandb_logger)

    # # run resnet
    main_resnet(wandb, wandb_logger)


