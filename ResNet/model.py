import torch
from torch import nn
from pytorch_lightning.core.module import LightningModule
from torch.nn import functional as F
from monai.networks.nets import resnet10
from torchvision.models import densenet121
import monai
from monai.networks.nets import DenseNet
torch.backends.cudnn.enabled = False
# import numpy as np
import torchmetrics
from torch.optim.lr_scheduler import StepLR


class ResNetModel(LightningModule):
    '''
    Resnet Model Class including the training, validation, and testing steps
    '''

    def __init__(self, class_weights):
        super().__init__()
        self.class_weights = class_weights

        self.net = nn.Sequential(
            DenseNet(spatial_dims=3, in_channels=1, out_channels=2),
            # densenet121(pretrained=False, spatial_dims=3, n_input_channels=1, num_classes=2),
            # resnet10(pretrained=False, spatial_dims=3, n_input_channels=1, num_classes=2),
            nn.Dropout(0.2)  # Adjust the dropout rate as needed
        )

        # Modify the classifier to match your output classes


        # self.net = resnet10(pretrained=False, spatial_dims=3, n_input_channels=1, num_classes=2)

        # # add a new fc layer
        # self.fc = nn.Linear(400, 5)

        # # combine the nets
        # self.net = nn.Sequential(self.resnet, self.fc)
        # self.loss = nn.CrossEntropyLoss()
        self.train_precision = torchmetrics.Precision(task='multiclass',num_classes=2, average='macro')
        self.train_recall = torchmetrics.Recall(task='multiclass',num_classes=2, average='macro')
        self.val_precision = torchmetrics.Precision(task='multiclass',num_classes=2, average='macro')
        self.val_recall = torchmetrics.Recall(task='multiclass',num_classes=2, average='macro')
        self.test_precision = torchmetrics.Precision(task='multiclass',num_classes=2, average='macro')
        self.test_recall = torchmetrics.Recall(task='multiclass',num_classes=2, average='macro')

        self.train_accuracy = torchmetrics.Accuracy(task='multiclass', average='macro', num_classes=2)

        self.train_macro_f1 = torchmetrics.classification.MulticlassF1Score(task='multiclass', num_classes=2, average='macro')

        self.train_auc = torchmetrics.classification.BinaryAUROC(task='multiclass', num_classes=2, average='macro')
        self.val_accuracy = torchmetrics.Accuracy(task='multiclass', average='macro', num_classes=2)

        self.val_macro_f1 = torchmetrics.classification.MulticlassF1Score(task='multiclass', num_classes=2, average='macro')

        self.val_auc = torchmetrics.classification.BinaryAUROC(task='multiclass', num_classes=2, average='macro')
        self.test_accuracy = torchmetrics.Accuracy(task='multiclass', average='macro', num_classes=2)

        self.test_macro_f1 = torchmetrics.classification.MulticlassF1Score(task='multiclass', num_classes=2, average='macro')

        self.test_auc = torchmetrics.classification.BinaryAUROC(task='multiclass', num_classes=2, average='macro')
        # self.loss = nn.BCEWithLogitsLoss(weight=torch.Tensor(class_weights) if class_weights else None)

        self.loss = nn.CrossEntropyLoss(weight=torch.Tensor(class_weights) if class_weights else None)
    def forward(self, x):
        out = self.net(x)
        # out = out.view(-1, 5)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, capturable=True)#, weight_decay=1e-3) # maybe higher learning rate bc of scheduler
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 35, 50], gamma=0.9)
        # lr_scheduler = {
        #     'scheduler': scheduler,
        #     'name': 'lr_logging'
        # }
        return [optimizer]
        # return [optimizer], [lr_scheduler]
    def calculate_class_weighted_accuracy(self, y_pred, y_true, class_weights):
        accuracy_per_class = (y_pred == y_true).float()

        # Apply class weights
        
        #     accuracy_per_class = accuracy_per_class.cuda()
        #     y_true = y_true.cuda()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if device.type == "cuda":
            weighted_accuracy_per_class = accuracy_per_class * torch.Tensor([class_weights[label] for label in y_true]).cuda()
        else: 
            accuracy_per_class = accuracy_per_class.cpu()
            weighted_accuracy_per_class = accuracy_per_class * torch.Tensor([class_weights[label] for label in y_true]).cpu()

        # Compute class_weighted accuracy
        class_weighted_acc = weighted_accuracy_per_class.sum() / len(y_true)

        return class_weighted_acc


    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.to(torch.long)
        x = torch.unsqueeze(x, 1)
        y_pred = self(x)
        y = torch.sub(y, 1)
        y = torch.where((y < 3), 0, y)
        y = torch.where((y >= 3) & (y <= 4), 1, y)
        


        # print("train predatory", y_pred)
        # print("prey", y)
        loss = self.loss(y_pred, y)
        # print("loss aquired")
        y_pred = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        # acc = 0#
        acc = (y_pred == y).float().mean()
        print(f'train y_pred argmax: {y_pred}')
        print(f'label: {y}')
        print("score", acc)

        # Log loss and accuracy
        # print(y)
        # class_weighted_acc = self.calculate_class_weighted_accuracy(y_pred, y, self.class_weights)
        self.log('train_loss', loss)
        self.log('train_acc', acc, prog_bar=True)

        self.train_accuracy(y_pred, y)
        self.train_macro_f1(y_pred, y)
        self.train_auc(y_pred, y)

        self.train_precision(y_pred, y)
        self.train_recall(y_pred, y)

        self.log('train_precision', self.train_precision, on_step=True, on_epoch=True)
        self.log('train_recall', self.train_recall, on_step=True, on_epoch=True)

        self.log('train_accuracy', self.train_accuracy, on_step=True, on_epoch=True)
        self.log('train_macro_f1', self.train_macro_f1, on_step=True, on_epoch=True)
        self.log('train_auc', self.train_auc, on_step=True, on_epoch=True)
        # self.log('train_class_weighted_acc', class_weighted_acc, prog_bar=True)
        # self.scheduler.step()

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y #.to(torch.long)
        y = torch.sub(y, 1)
        # print(f'label after sub: {y}')
        y = torch.where((y < 3), 0, y)
        y = torch.where((y >= 3) & (y <= 4), 1, y)
        
        x = torch.unsqueeze(x, 1)
        # print(f'input batch shape: {x.shape}')
        # print(f'label batch shape: {y.shape}')

        y_pred = self(x)
        # print(f'model output shape: {y_pred.shape}')

        # print("cal predatory", y_pred)
        # print("prey", y)
        loss = self.loss(y_pred, y)
        # print("loss aquired")
        y_pred = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        # print(f'argmax output shape: {y_pred.shape}')
        # y_pred += 1
        acc = (y_pred == y).float().mean()
        print(f'y_pred argmax: {y_pred}')
        print(f'label: {y}')
        print("score", acc)

        # acc = np.sum(y_pred == y).float().mean() * 100.0
        # print(f'accuracy: {acc}')

        # Log loss and accuracy
        self.log('val_loss', loss)
        self.log('val_acc', acc, prog_bar=True)
        # class_weighted_acc = self.calculate_class_weighted_accuracy(y_pred, y, self.class_weights)
        self.val_accuracy(y_pred, y)
        self.val_macro_f1(y_pred, y)
        self.val_auc(y_pred, y)

        self.log('val_accuracy', self.val_accuracy, on_step=True, on_epoch=True)
        self.log('val_macro_f1', self.val_macro_f1, on_step=True, on_epoch=True)
        self.log('val_auc', self.val_auc, on_step=True, on_epoch=True)

        self.val_precision(y_pred, y)
        self.val_recall(y_pred, y)

        self.log('val_precision', self.val_precision, on_step=True, on_epoch=True)
        self.log('val_recall', self.val_recall, on_step=True, on_epoch=True)
        # self.log('val_class_weighted_acc', class_weighted_acc, prog_bar=True)
        

        
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        # print(f'input bacth shape: {x.shape}')
        # print(f'label batch shape: {y.shape}')
        y = y.to(torch.long)

        y_pred = self(x)
        # print(f'model outpit shape: {y_pred.shape}')
        y = torch.sub(y, 1)
        y = torch.where((y < 3), 0, y)
        y = torch.where((y >= 3) & (y <= 4), 1, y)
        

        # print(" test predatory", y_pred)
        # print("prey", y)
        loss = self.loss(y_pred, y)
        # print("loss aquired")
        y_pred = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)

        acc = (y_pred == y).float().mean()
        # print(f'y_pred argmax: {y_pred}')
        # print(f'label: {y}')
        # print("score", acc)

        # Log loss and accuracy
        # class_weighted_acc = self.calculate_class_weighted_accuracy(y_pred, y, self.class_weights)
        # self.log('test_class_weighted_acc', class_weighted_acc, prog_bar=True)
        self.log('test_loss', loss)
        self.log('test_acc', acc, prog_bar=True)
        self.test_accuracy(y_pred, y)
        self.test_macro_f1(y_pred, y)
        self.test_auc(y_pred, y)

        self.log('test_accuracy', self.test_accuracy, on_step=True, on_epoch=True)
        self.log('test_macro_f1', self.test_macro_f1, on_step=True, on_epoch=True)
        self.log('test_auc', self.test_auc, on_step=True, on_epoch=True)

        self.test_precision(y_pred, y)
        self.test_recall(y_pred, y)

        self.log('test_precision', self.test_precision, on_step=True, on_epoch=True)
        self.log('test_recall', self.test_recall, on_step=True, on_epoch=True)

        return loss

