import wandb
import os
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor

from conv3D.model import AdniModel
# from dataset import odule
from multimodal_dataset import NCANDADataModule
from multimodal_dataset_triplet import NCANDADataTripletModule

from ResNet.model import ResNetModel
from multimodal.model import MultiModModel
from multimodal.model_language import MultiModModelWithLanguage
from multimodal.daft_model import DAFTModel
from multimodal.center_model import CenterModel
from multimodal.triplet_model import TripletModel

def main_conv3d(wandb, wandb_logger):
    '''
    main function to run the conv3d architecture
    '''
    # ge tthe model
    model = AdniModel()

    # load the data
    data = NCANDADataModule()

    # Optional
    wandb.watch(model, log="all")

    # train the network
    trainer = Trainer(max_epochs=15, logger=wandb_logger, log_every_n_steps=1, accelerator='gpu', devices=1)
    trainer.fit(model, data)


def main_resnet(wandb, wandb_logger):
    '''
    main function to run the resnet architecture
    '''
    # ge the model
    model = ResNetModel()

    # load the data
    data = NCANDADataModule()

    # Optional
    wandb.watch(model, log="all")

    # train the network
    trainer = Trainer(max_epochs=15, logger=wandb_logger, log_every_n_steps=1, accelerator='gpu', devices=1)
    trainer.fit(model, data)


def main_multimodal(wandb, wandb_logger):
    '''
    main function to run the multimodal architecture
    
    
    '''

    seed_everything(23)
    
    # load the data
    data = NCANDADataModule()

    data.prepare_data()

    train_loader = data.train_dataloader()

    val_loader = data.val_dataloader()

    # ge the model
    model = MultiModModel(class_weight=data.class_weight, scaler=data.scaler)

    # Optional
    wandb.watch(model, log="all")
    
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # train the network
    trainer = Trainer(max_epochs=60, logger=wandb_logger, log_every_n_steps=1, accelerator='gpu', devices=1, callbacks=[lr_monitor])
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


def main_daft(wandb, wandb_logger):
    '''
    main function to run the multimodal architecture
    '''
    seed_everything(23)
    # load the data
    data = NCANDADataModule()

    data.prepare_data()

    train_loader = data.train_dataloader()

    val_loader = data.val_dataloader()

    # ge the model
    model = DAFTModel(class_weight=data.class_weight, scaler=data.scaler)

    # Optional
    wandb.watch(model, log="all")

    lr_monitor = LearningRateMonitor(logging_interval='step')

    # train the network
    trainer = Trainer(max_epochs=30, logger=wandb_logger, log_every_n_steps=1, callbacks=[lr_monitor], accelerator='gpu', devices=1)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    
def main_language(wandb, wandb_logger):
    '''
    main function to run the multimodal architecture
    '''
    seed_everything(23)
    # load the data
    data = NCANDADataModule()

    data.prepare_data()

    train_loader = data.train_dataloader()

    val_loader = data.val_dataloader()

    # ge the model
    model = MultiModModelWithLanguage(class_weight=data.class_weight, scaler=data.scaler)

    # Optional
    wandb.watch(model, log="all")

    lr_monitor = LearningRateMonitor(logging_interval='step')

    # train the network
    trainer = Trainer(max_epochs=60, logger=wandb_logger, log_every_n_steps=1, callbacks=[lr_monitor], accelerator='gpu', devices=1)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    

def main_center(wandb, wandb_logger):
    '''
    main function to run the multimodal architecture
    '''
    seed_everything(23)
    # load the data
    data = NCANDADataModule()

    data.prepare_data()

    train_loader = data.train_dataloader()

    val_loader = data.val_dataloader()

    # ge the model
    model = CenterModel(class_weight=data.class_weight, scaler=data.scaler)

    # Optional
    wandb.watch(model, log="all")

    lr_monitor = LearningRateMonitor(logging_interval='step')

    # train the network
    trainer = Trainer(max_epochs=35, logger=wandb_logger, log_every_n_steps=1, callbacks=[lr_monitor], accelerator='gpu', devices=1)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    
def main_triplet(wandb, wandb_logger):
    '''
    main function to run the multimodal architecture
    '''
    seed_everything(23)
    # load the data
    data = NCANDADataTripletModule()

    data.prepare_data()

    train_loader = data.train_dataloader()

    val_loader = data.val_dataloader()

    # ge the model
    model = TripletModel(class_weight=data.class_weight, scaler=data.scaler)

    # Optional
    wandb.watch(model, log="all")

    lr_monitor = LearningRateMonitor(logging_interval='step')

    # train the network
    trainer = Trainer(max_epochs=60, logger=wandb_logger, log_every_n_steps=1, callbacks=[lr_monitor], accelerator='gpu', devices=1)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == '__main__':

    # create wandb objects to track runs
    # wandb.init(project="ncanda-imaging")
    # wandb.config = {
    #     "learning_rate": 1e-4,
    #     "epochs": 5,
    #     "batch_size": 1
    # }

    wandb_logger = WandbLogger(wandb.init(project="ncanda-imaging", entity="magda"))

    # # run conv3d
    # main_conv3d(wandb, wandb_logger)

    # run resnet
    # main_resnet(wandb, wandb_logger)

    # run multimodal
    # main_multimodal(wandb, wandb_logger)
    
    # run daft model
    # main_daft(wandb, wandb_logger)
    
    # run model with language
    # main_language(wandb, wandb_logger)
    
    # run model with bce + center loss
    # main_center(wandb, wandb_logger)
    
    # run model with bce + center loss + triplet loss
    main_triplet(wandb, wandb_logger)
