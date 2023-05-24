import torch
from torch import nn
from pytorch_lightning.core.module import LightningModule
from torch.nn import functional as F
from monai.networks.nets import resnet10, resnet18, resnet34, resnet50
from monai.networks.nets.swin_unetr import SwinUNETR
from monai.networks.blocks import UnetrBasicBlock
from notes import CustomSwinEncoder
from settings import IMAGE_SIZE, FEATURES, BATCH_SIZE, TARGET, NUM_FEATURES
import torchmetrics
import pandas as pd


class MultiModModel(LightningModule):
    '''
    Resnet Model Class including the training, validation and testing steps
    '''

    def __init__(self, class_weight=None, scaler=None):

        super().__init__()
 
        if class_weight is None:
            data = NCANDADataModule()
            data.prepare_data()
            class_weight=data.class_weight
        if scaler is None:
            data = NCANDADataModule()
            data.prepare_data()
            scaler=data.scaler
 
        self.class_weight = class_weight

        self.scaler = scaler

        self.save_hyperparameters()

        self.view_maps = nn.Identity()

        self.resnet = resnet10(pretrained=False,
                            spatial_dims=3,
                            num_classes=120,
                             n_input_channels=1
                             )
        #self.swin_tf = SwinUNETR(
        #    img_size=IMAGE_SIZE,
        #    in_channels=1,
        #    out_channels=1,
        #    feature_size=12,  # feature size should be divisible by 12
        #)
        #print("Initialized swin UNETR!")
        # use pre-trained weights
        #weight = torch.load("./model_swinvit.pt")
        
        #self.swin_enc = CustomSwinEncoder(
        #    img_size=IMAGE_SIZE,
        #    in_channels=1,
        #    out_channels=1,
        #    feature_size=12,  # feature size should be divisible by 12
        #)

        #self.swin_class_layer = nn.Linear(24576, 120)

        self.NUM_FEATURES = NUM_FEATURES

        # fc layer for tabular data
        #self.fc1 = nn.Linear(self.NUM_FEATURES, 120)

        # first fc layer which takes concatenated imput
        #self.fc2 = nn.Linear((120 + 120), 32)
        self.fc2 = nn.Linear(120, 32)        

        # final fc layer which takes concatenated imput
        self.fc3 = nn.Linear(32, 1)
        
        self.train_macro_accuracy = torchmetrics.Accuracy(task='multiclass', average='macro', num_classes=2)
        
        self.val_macro_accuracy = torchmetrics.Accuracy(task='multiclass', average='macro', num_classes=2)
        
        self.test_macro_accuracy = torchmetrics.Accuracy(task='multiclass', average='macro', num_classes=2)

        self.train_accuracy = torchmetrics.Accuracy(task='multiclass', average='micro', num_classes=2)
        
        self.val_accuracy = torchmetrics.Accuracy(task='multiclass', average='micro', num_classes=2)
        
        self.test_accuracy = torchmetrics.Accuracy(task='multiclass', average='micro', num_classes=2)
        
        self.results_column_names = ['subject', 'label', 'prediction', 'age', 'sex']

        self.train_results_df = pd.DataFrame(columns=self.results_column_names)

        self.train_results_df_all = pd.DataFrame(columns=self.results_column_names)

        self.val_results_df_all = pd.DataFrame(columns=self.results_column_names)

        self.val_results_df = pd.DataFrame(columns=self.results_column_names)

    def forward(self, img, tab=None):
        """

        x is the input data

        """
        # run the model for the image
        img = torch.unsqueeze(img, 1)
        img = img.to(torch.float32)
        
        # to view attention maps
        img = img.reshape((img.shape[0], IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE))
        img = self.view_maps(img)
        img = img.reshape((img.shape[0], 1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE)) 
        

        print(img.shape, "before resnet the shape is..") 
        img = self.resnet(img)
        print("aftre resnet the shape is ...", img.shape)
        img = torch.flatten(img, start_dim=1) 

        # change the dtype of the tabular data
        #tab = tab.to(torch.float32)

        # forward tabular data
        #tab = F.relu(self.fc1(tab))
        
        # concat image and tabular data
        #x = torch.cat((img, tab), dim=1)
        x = img        

        x = F.relu(self.fc2(x))

        out = self.fc3(x)

        out = torch.squeeze(out)

        return out

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4) #1e-3
        
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,35,50], gamma=0.8)

        lr_scheduler = {
            'scheduler': scheduler,
            'name': 'lr_logging'
        }

        return [optimizer], [lr_scheduler]
        

    def training_step(self, batch, batch_idx):

        img, tab, y, subject_id = batch
        
        img = torch.tensor(img).float()
        
        y = y.to(torch.float32)

        y_pred = self(img, tab)

        loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.class_weight).float())

        loss = loss_func(y_pred, y.squeeze())
        
        y_pred_tag = torch.round(torch.sigmoid(y_pred))
        
        self.train_results_df['subject'] = tuple(subject_id)
        self.train_results_df['label'] = y.squeeze().detach().cpu().numpy()
        self.train_results_df['prediction'] = y_pred_tag.detach().cpu().numpy()

        tab_bef_normalization = self.scaler.inverse_transform(tab.detach().cpu().numpy())
        self.train_results_df['age'] = tab_bef_normalization[:,2]
        self.train_results_df['sex'] = tab_bef_normalization[:, 1]
        
        self.train_results_df_all = pd.concat([self.train_results_df_all , self.train_results_df], ignore_index=True)
        
        if BATCH_SIZE == 1:
            self.train_accuracy(torch.unsqueeze(y_pred_tag, 0), y)
            
            self.train_macro_accuracy(torch.unsqueeze(y_pred_tag, 0), y)
        else:
            self.train_accuracy(y_pred_tag, y)
            
            self.train_macro_accuracy(y_pred_tag, y)
        
        self.log('train_acc_step', self.train_accuracy, on_step=False, on_epoch=True)
        self.log('train_macro_acc_step', self.train_macro_accuracy, on_step=True, on_epoch=True)
        # Log loss
        self.log('train_loss', loss, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):

        img, tab, y, subject_id = batch
        y = y.to(torch.float32)

        y_pred = self(img, tab)

        loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.class_weight).float())
        
        loss = loss_func(y_pred, y.squeeze())
        y_pred_tag = torch.round(torch.sigmoid(y_pred))
        
        self.val_results_df['subject'] = tuple(subject_id)
        self.val_results_df['label'] = y.squeeze().detach().cpu().numpy()
        self.val_results_df['prediction'] = y_pred_tag.detach().cpu().numpy()
        
        tab_bef_normalization = self.scaler.inverse_transform(tab.detach().cpu().numpy())
        self.val_results_df['age'] = tab_bef_normalization[:,2]
        self.val_results_df['sex'] = tab_bef_normalization[:, 1]
        
        self.val_results_df_all = pd.concat([self.val_results_df_all , self.val_results_df], ignore_index=True)
        
        if BATCH_SIZE == 1:
            
            self.val_accuracy(torch.unsqueeze(y_pred_tag, 0), y)
            
            self.val_macro_accuracy(torch.unsqueeze(y_pred_tag, 0), y)
        else:
            self.val_accuracy(y_pred_tag, y)
            
            self.val_macro_accuracy(y_pred_tag, y)
        
        self.log('val_acc_step', self.val_accuracy, on_step=False, on_epoch=True)
        self.log('val_macro_acc_step', self.val_macro_accuracy, on_step=True, on_epoch=True)

        # Log loss
        self.log('val_loss', loss, on_step=True, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):

        img, tab, y, subject_id = batch
        y = y.to(torch.float32)

        y_pred = self(img, tab)

        loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.class_weight).float())

        loss = loss_func(y_pred, y.squeeze())        # loss = F.binary_cross_entropy(torch.sigmoid(y_pred), y.squeeze(), pos_weights = )
        
        y_pred_tag = torch.round(torch.sigmoid(y_pred))
                
        if BATCH_SIZE == 1:
            
            self.test_accuracy(torch.unsqueeze(y_pred_tag, 0), y)
            
            self.test_macro_accuracy(torch.unsqueeze(y_pred_tag, 0), y)
        else:
            self.test_accuracy(y_pred_tag, y)
            
            self.test_macro_accuracy(y_pred_tag, y)
        
        self.log('test_acc_step', self.test_accuracy, on_step=True, on_epoch=False)
        self.log('test_macro_acc_step', self.test_macro_accuracy, on_step=True, on_epoch=True)
        
        self.log("test loss", loss)

        return loss
        
    
    def training_epoch_end(self, outs):
        
        filename_out = '/home/users/tulikaj/results/train_out_' + str(self.current_epoch) + '_' + TARGET + '_' + self.trainer.logger.experiment.name + '.csv'

        self.train_results_df_all.to_csv(filename_out)

        # Clear the dataframe so the new epoch can start fresh
        self.train_results_df_all = pd.DataFrame(columns=self.results_column_names)
        # log epoch metric
        self.log('train_acc_epoch', self.train_accuracy)
        self.log('train_macro_acc_epoch', self.train_macro_accuracy)
        
    
    def validation_epoch_end(self, outputs):
        # log epoch metric

        filename_out = '/home/users/tulikaj/results/val_out_' + str(self.current_epoch) + '_' + TARGET + '_' + self.trainer.logger.experiment.name + '.csv'
        
        self.val_results_df_all.to_csv(filename_out)

        # Clear the dataframe so the new epoch can start fresh
        self.val_results_df_all = pd.DataFrame(columns=self.results_column_names)

        self.log('val_acc_epoch', self.val_accuracy)
        self.log('val_macro_acc_epoch', self.val_macro_accuracy)
        
