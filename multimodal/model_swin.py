import torch
from torch import nn
from pytorch_lightning.core.module import LightningModule
from torch.nn import functional as F
from monai.networks.nets.swin_unetr import SwinUNETR
from monai.networks.blocks import UnetrBasicBlock
from notes import CustomSwinEncoder
from settings import IMAGE_SIZE, FEATURES, BATCH_SIZE, TARGET, NUM_FEATURES
import torchmetrics
import pandas as pd
from torchmetrics.classification import MulticlassAccuracy


USE_TAB_DATA = 1
CROSS_CORR = 1

class MultiModModelSwinEnc(LightningModule):
    '''
    Swin Encoder Model Class including the training, validation and testing steps
    '''

    def __init__(self, class_weight, scaler):

        super().__init__()

        self.class_weight = class_weight

        self.scaler = scaler

        self.save_hyperparameters()

        """
        self.swin_enc = CustomSwinEncoder(
            img_size=IMAGE_SIZE,
            in_channels=1,
            out_channels=1,
            feature_size=12,
        )
        """
        self.swin_enc = SwinUNETR(
               img_size=IMAGE_SIZE,
               in_channels=1,
               out_channels=1,
               feature_size=12,  # feature size should be divisible by 12
            )
        #weight = torch.load("./model_swinvit.pt")
        #self.swin_enc.load_from(weights=weight)
        #print("Using pretrained self-supervied Swin UNETR backbone weights !")

        # self.swin_fc_layer = nn.Linear(24576, 120)
        # self.swin_fc_layer = nn.Linear(98304, 120)
        self.post_swin_conv_0, self.post_swin_relu_0 = None, None
        out_channels_0 = 1
        if IMAGE_SIZE == 128:
            self.post_swin_conv_0 = nn.Conv3d(
                in_channels=1,
                out_channels=60,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=True
            )
            self.post_swin_relu_0 = nn.ReLU()
            out_channels_0 = 60        

        self.post_swin_conv_1 = nn.Conv3d(
            in_channels=out_channels_0,
            out_channels=30,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=True
        )
        self.post_swin_relu_1 = nn.ReLU()
        self.post_swin_maxpool_1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.post_swin_conv_2 = nn.Conv3d(
            in_channels=30,
            out_channels=60,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=True
        )
        self.post_swin_relu_2 = nn.ReLU()
        self.post_swin_maxpool_2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.post_swin_conv_3 = nn.Conv3d(
            in_channels=60,
            out_channels=120,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=True
        )
        self.post_swin_relu_3 = nn.ReLU()
        self.post_swin_maxpool_3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.NUM_FEATURES = NUM_FEATURES

        if USE_TAB_DATA:
            # fc layer for tabular data
            self.fc1 = nn.Linear(self.NUM_FEATURES, 120)
        
        self.fc2 = nn.Linear(120, 32)

        if USE_TAB_DATA:
            if CROSS_CORR:
                self.fc3 = nn.Linear(32*2 - 1, 1)
            else:
                # final fc layer which takes concatenated input
                self.fc3 = nn.Linear(64, 1)
        else:
            self.fc3 = nn.Linear(32, 1)

        self.train_macro_accuracy = torchmetrics.Accuracy(task='multiclass', average='macro', num_classes=2)

        self.train_F1_score = torchmetrics.F1Score(task='multiclass', average='macro', num_classes=2)

        self.train_AUROC = torchmetrics.classification.BinaryAUROC(thresholds=None)

        self.val_macro_accuracy = torchmetrics.Accuracy(task='multiclass', average='macro', num_classes=2)

        self.val_F1_score = torchmetrics.F1Score(task='multiclass', num_classes=2)

        self.val_AUROC = torchmetrics.classification.BinaryAUROC(thresholds=None)

        self.test_macro_accuracy = torchmetrics.Accuracy(task='multiclass', average='macro', num_classes=2)

        self.test_F1_score = torchmetrics.F1Score(task='multiclass', num_classes=2)

        self.test_AUROC = torchmetrics.classification.BinaryAUROC(thresholds=None)

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
        # print("image shape, forward pass initially", img.shape)

        img = torch.unsqueeze(img, 1)
        img = img.to(torch.float32)

        img = self.swin_enc(img)
        print("image shape after swin encoder", img.shape)

        # img = torch.flatten(img, start_dim=1)
        # print("image shap after flattening, before swin fc layer", img.shape)
        # img = self.swin_fc_layer(img)
        
        if IMAGE_SIZE == 128:
            img = self.post_swin_conv_0(img)
            img = self.post_swin_relu_0(img)

        img = self.post_swin_conv_1(img)
        img = self.post_swin_relu_1(img)
        img = self.post_swin_maxpool_1(img)
        img = self.post_swin_conv_2(img)
        img = self.post_swin_relu_2(img)
        img = self.post_swin_maxpool_2(img)
        img = self.post_swin_conv_3(img)
        img = self.post_swin_relu_3(img)
        img = self.post_swin_maxpool_3(img)

        img = torch.flatten(img, start_dim=1)

        if USE_TAB_DATA and tab is not None:
            # change the dtype of the tabular data
            tab = tab.to(torch.float32)

            # forward tabular data
            tab = F.relu(self.fc1(tab))
            
            # forward tab through shared fc layer
            tab = F.relu(self.fc2(tab))
            
            # forward image through shared fc layer
            img = F.relu(self.fc2(img))

            if CROSS_CORR:
                img = img.unsqueeze(0)
                tab = tab.unsqueeze(1)
                x = F.conv1d(img, tab, padding=32-1, groups=img.size(1))
                x = x.squeeze()
            else:
                # concat image and tabular data
                x = torch.cat((img, tab), dim=1)

        elif USE_TAB_DATA:
            tab = torch.ones((1, self.NUM_FEATURES), dtype=torch.float32)
            tab = F.relu(self.fc1(tab))
            tab = F.relu(self.fc2(tab))
            img = F.relu(self.fc2(img))
            
            img = img.unsqueeze(0)
            tab = tab.unsqueeze(1)
            x = F.conv1d(img, tab, padding=32-1, groups=img.size(1))
            x = x.squeeze()
        else:
            x = F.relu(self.fc2(img))

        # forward concatenated feature vector through another fc layer
        out = self.fc3(x)

        out = torch.squeeze(out)

        return out

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)  # 1e-3

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 35, 50], gamma=0.8)

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
        self.train_results_df['age'] = tab_bef_normalization[:, 2]
        self.train_results_df['sex'] = tab_bef_normalization[:, 1]

        self.train_results_df_all = pd.concat([self.train_results_df_all, self.train_results_df], ignore_index=True)

        if BATCH_SIZE == 1:
            self.train_accuracy(torch.unsqueeze(y_pred_tag, 0), y)

            self.train_macro_accuracy(torch.unsqueeze(y_pred_tag, 0), y)

            self.train_F1_score(torch.unsqueeze(y_pred_tag, 0), y)

            self.train_AUROC(torch.unsqueeze(y_pred_tag, 0), y)
        else:
            self.train_accuracy(y_pred_tag, y)

            self.train_macro_accuracy(y_pred_tag, y)

            self.train_F1_score(y_pred_tag, y)

            self.train_AUROC(y_pred_tag, y)

        self.log('train_acc_step', self.train_accuracy, on_step=False, on_epoch=True)
        self.log('train_macro_acc_step', self.train_macro_accuracy, on_step=True, on_epoch=True)
        self.log('train_F1_score', self.train_F1_score, on_step=True, on_epoch=True)
        self.log('train_AUROC', self.train_AUROC, on_step=True, on_epoch=True)
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
        self.val_results_df['age'] = tab_bef_normalization[:, 2]
        self.val_results_df['sex'] = tab_bef_normalization[:, 1]

        self.val_results_df_all = pd.concat([self.val_results_df_all, self.val_results_df], ignore_index=True)

        if BATCH_SIZE == 1:

            self.val_accuracy(torch.unsqueeze(y_pred_tag, 0), y)

            self.val_macro_accuracy(torch.unsqueeze(y_pred_tag, 0), y)

            self.val_F1_score(torch.unsqueeze(y_pred_tag, 0), y)

            self.val_AUROC(torch.unsqueeze(y_pred_tag, 0), y)
        else:
            self.val_accuracy(y_pred_tag, y)

            print("computing val_macro_acc_epoch for step epoch right now...")
            print(f"y_pred_tag = {y_pred_tag}")
            print(f"y = {y}")
            self.val_macro_accuracy(y_pred_tag, y)

            self.val_F1_score(y_pred_tag, y)

            self.val_AUROC(y_pred_tag, y)

        self.log('val_acc_step', self.val_accuracy, on_step=False, on_epoch=True)
        self.log('val_macro_acc_step', self.val_macro_accuracy, on_step=True, on_epoch=True)
        self.log('val_F1_score', self.val_F1_score, on_step=True, on_epoch=True)
        self.log('val_AUROC', self.val_AUROC, on_step=True, on_epoch=True)

        # Log loss
        self.log('val_loss', loss, on_step=True, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):

        img, tab, y, subject_id = batch
        y = y.to(torch.float32)

        y_pred = self(img, tab)

        loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.class_weight).float())

        loss = loss_func(y_pred, y.squeeze())
        y_pred_tag = torch.round(torch.sigmoid(y_pred))

        if BATCH_SIZE == 1:
            self.test_accuracy(torch.unsqueeze(y_pred_tag, 0), y)

            self.test_macro_accuracy(torch.unsqueeze(y_pred_tag, 0), y)

            self.test_F1_score(torch.unsqueeze(y_pred_tag, 0), y)

            self.test_AUROC(torch.unsqueeze(y_pred_tag, 0), y)
        else:
            self.test_accuracy(y_pred_tag, y)

            self.test_macro_accuracy(y_pred_tag, y)

            self.test_F1_score(y_pred_tag, y)

            self.test_AUROC(y_pred_tag, y)

        self.log('test_acc_step', self.test_accuracy, on_step=True, on_epoch=False)
        self.log('test_macro_acc_step', self.test_macro_accuracy, on_step=True, on_epoch=True)
        self.log('test_F1_score', self.test_F1_score, on_step=True, on_epoch=True)
        self.log('test_AUROC', self.test_AUROC, on_step=True, on_epoch=True)

        self.log("test loss", loss)

        return loss

    def training_epoch_end(self, outs):

        filename_out = '/home/users/tulikaj/results/train_out_' + str(
            self.current_epoch) + '_' + TARGET + '_' + self.trainer.logger.experiment.name + '.csv'

        self.train_results_df_all.to_csv(filename_out)

        # Clear the dataframe so the new epoch can start fresh
        self.train_results_df_all = pd.DataFrame(columns=self.results_column_names)
        # log epoch metric
        self.log('train_acc_epoch', self.train_accuracy)
        self.log('train_macro_acc_epoch', self.train_macro_accuracy)

    def validation_epoch_end(self, outputs):
        # log epoch metric

        filename_out = '/home/users/tulikaj/results/val_out_' + str(
            self.current_epoch) + '_' + TARGET + '_' + self.trainer.logger.experiment.name + '.csv'

        self.val_results_df_all.to_csv(filename_out)

        # Clear the dataframe so the new epoch can start fresh
        self.val_results_df_all = pd.DataFrame(columns=self.results_column_names)

        self.log('val_acc_epoch', self.val_accuracy)
        self.log('val_macro_acc_epoch', self.val_macro_accuracy)

    def test_epoch_end(self, outputs):
        # log test set evaluation metrics
        self.log('test_acc_epoch', self.test_accuracy)
        self.log('test_macro_acc_epoch', self.test_macro_accuracy)
