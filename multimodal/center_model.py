import torch
from torch import nn
from pytorch_lightning.core.module import LightningModule
from torch.nn import functional as F
from monai.networks.nets import resnet10, resnet18, resnet34, resnet50
from settings import IMAGE_SIZE, FEATURES, BATCH_SIZE, TARGET
import torchmetrics
import pandas as pd


class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss


class CenterModel(LightningModule):
    '''
    Resnet Model Class including the training, validation and testing steps
    '''

    def __init__(self, class_weight, scaler):

        super().__init__()
        
        self.class_weight = class_weight

        self.scaler = scaler

        self.resnet = resnet10(pretrained=False,
                              spatial_dims=3,
                              num_classes=len(FEATURES),
                              n_input_channels=1
                              )

        self.NUM_FEATURES = len(FEATURES)

        # fc layer only tabular data
        self.fc1 = nn.Linear(self.NUM_FEATURES, 128)
        
        # fc layer only imaging data
        self.fc2 = nn.Linear(self.NUM_FEATURES, 128)
        
        # Shared Layer
        self.fc3 = nn.Linear(128, 128)

        # first fc layer which takes concatenated imput
        self.fc4 = nn.Linear((128 + 128), 64)
        
        # final fc layer which takes concatenated imput
        self.fc5 = nn.Linear(64, 1)
        
        self.center_loss = CenterLoss(num_classes=2, feat_dim=64, use_gpu=True)

        self.bce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.class_weight).float())

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


    def forward(self, img, tab):
        """

        x is the input data

        """
        # run the model for the image

        # print(img.shape)
        img = torch.unsqueeze(img, 1)
        img = img.to(torch.float32)
        # print(img.type)
        # print(img.shape)
        
        img = self.resnet(img)

        # change the dtype of the tabular data
        tab = tab.to(torch.float32)

        # forward tabular data
        tab = F.relu(self.fc1(tab))
        
        img = F.relu(self.fc2(img))
        
        tab = F.relu(self.fc3(tab))
        
        img = F.relu(self.fc3(img))
        
        # concat image and tabular data
        x = torch.cat((img, tab), dim=1)

        x_feats = self.fc4(x)
        
        x = F.relu(x_feats)

        out = self.fc5(x)

        out = torch.squeeze(out)

        return out, x_feats

    def configure_optimizers(self):
        
        # optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        
        # Train with different learning rates - scheduler is not doing anything in that case
        my_list = ['center_loss.centers']
        center_params = list(filter(lambda kv: kv[0] in my_list, self.named_parameters()))
        model_params = list(filter(lambda kv: kv[0] not in my_list, self.named_parameters()))

        optimizer = torch.optim.Adam([
            {'params': [temp[1] for temp in model_params]},
            {'params': center_params[0][1], 'lr': 1e-4}
        ], lr=1e-3)
        
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,35,50], gamma=0.8)

        #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=[0.0001, 0.00001], max_lr=[0.001, 0.0001], mode='triangular2',cycle_momentum=False)

        lr_scheduler = {
            'scheduler': scheduler,
            'name': 'lr_logging'
        }

        return [optimizer], [lr_scheduler]
        

    def training_step(self, batch, batch_idx):

        img, tab, y, subject_id = batch
        
        img = img.clone().detach().float()
        # img = torch.tensor(img).float()
        
        y = y.to(torch.float32)

        y_pred, x_feats = self(img, tab)
        
        bce_loss_f = self.bce_loss(y_pred, y.squeeze())

        center_loss_f = self.center_loss(x_feats, y.squeeze())

        loss = 1.0*bce_loss_f + 0.2*center_loss_f

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
        self.log('train_bce_loss', bce_loss_f, on_step=True, on_epoch=True)
        self.log('train_center_loss', center_loss_f, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):

        img, tab, y, subject_id = batch
        
        img = img.clone().detach().float()
        
        y = y.to(torch.float32)

        y_pred, x_feats = self(img, tab)

        bce_loss_f = self.bce_loss(y_pred, y.squeeze())

        center_loss_f = self.center_loss(x_feats, y.squeeze())

        loss = 1.0*bce_loss_f + 0.2*center_loss_f
        
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
        self.log('val_bce_loss', bce_loss_f, on_step=True, on_epoch=True)
        self.log('val_center_loss', center_loss_f, on_step=True, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):

        img, tab, y, subject_id = batch
        
        img = img.clone().detach().float()
                
        y = y.to(torch.float32)

        y_pred, x_feats = self(img, tab)

        bce_loss_f = self.bce_loss(y_pred, y.squeeze())

        center_loss_f = self.center_loss(x_feats, y.squeeze())

        loss = 1.0*bce_loss_f + 0.2*center_loss_f

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
        self.log('test_bce_loss', bce_loss_f, on_step=True, on_epoch=True)
        self.log('test_center_loss', center_loss_f, on_step=True, on_epoch=True)

        return loss
        
    
    def training_epoch_end(self, outs):
        
        filename_out = '/home/users/paschali/results/train_out_center_02_64' + str(self.current_epoch) + '_' + TARGET + '_' + self.trainer.logger.experiment.name + '.csv'

        self.train_results_df_all.to_csv(filename_out)

        # Clear the dataframe so the new epoch can start fresh
        self.train_results_df_all = pd.DataFrame(columns=self.results_column_names)
        # log epoch metric
        self.log('train_acc_epoch', self.train_accuracy)
        self.log('train_macro_acc_epoch', self.train_macro_accuracy)
        
    
    def validation_epoch_end(self, outputs):
        # log epoch metric

        filename_out = '/home/users/paschali/results/val_out_center_02_64' + str(self.current_epoch) + '_' + TARGET + '_' + self.trainer.logger.experiment.name + '.csv'
        
        self.val_results_df_all.to_csv(filename_out)

        # Clear the dataframe so the new epoch can start fresh
        self.val_results_df_all = pd.DataFrame(columns=self.results_column_names)

        self.log('val_acc_epoch', self.val_accuracy)
        self.log('val_macro_acc_epoch', self.val_macro_accuracy)
        
