import torch
from torch import nn
from pytorch_lightning.core.module import LightningModule
from torch.nn import functional as F
from monai.networks.nets import resnet10
torch.backends.cudnn.enabled = False
import numpy as np


class ResNetModel(LightningModule):
    '''
    Resnet Model Class including the training, validation, and testing steps
    '''

    def __init__(self, class_weights):
        super().__init__()
        self.class_weights = class_weights

        self.net = resnet10(pretrained=False, spatial_dims=3, n_input_channels=1, num_classes=5)

        # # add a new fc layer
        # self.fc = nn.Linear(400, 5)

        # # combine the nets
        # self.net = nn.Sequential(self.resnet, self.fc)
        # self.loss = nn.CrossEntropyLoss()
        self.loss = nn.CrossEntropyLoss(weight=torch.Tensor(class_weights.values()) if class_weights else None)
    def forward(self, x):
        out = self.net(x)
        # out = out.view(-1, 5)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, capturable=True)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.to(torch.long)
        x = torch.unsqueeze(x, 1)
        y_pred = self(x)
        y = torch.sub(y, 1)

        print("train predatory", y_pred)
        print("prey", y)
        loss = F.cross_entropy(y_pred, y)
        print("loss aquired")
        y_pred = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        # acc = 0#
        acc = (y_pred == y).float().mean()
        print(f'y_pred argmax: {y_pred}')
        print(f'label: {y}')
        print("score", acc)

        # Log loss and accuracy
        print(y)
        self.log('train_loss', loss)
        self.log('train_acc', acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y #.to(torch.long)
        y = torch.sub(y, 1)
        print(f'label after sub: {y}')
        x = torch.unsqueeze(x, 1)
        print(f'input batch shape: {x.shape}')
        print(f'label batch shape: {y.shape}')

        y_pred = self(x)
        print(f'model output shape: {y_pred.shape}')

        print("cal predatory", y_pred)
        print("prey", y)
        loss = self.loss(y_pred, y)
        print("loss aquired")
        y_pred = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        print(f'argmax output shape: {y_pred.shape}')
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

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        print(f'input bacth shape: {x.shape}')
        print(f'label batch shape: {y.shape}')
        y = y.to(torch.long)

        y_pred = self(x)
        print(f'model outpit shape: {y_pred.shape}')
        y = torch.sub(y, 1)

        print(" test predatory", y_pred)
        print("prey", y)
        loss = F.cross_entropy(y_pred, y)
        print("loss aquired")
        y_pred = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)

        acc = (y_pred == y).float().mean()
        print(f'y_pred argmax: {y_pred}')
        print(f'label: {y}')
        print("score", acc)

        # Log loss and accuracy
        self.log('test_loss', loss)
        self.log('test_acc', acc, prog_bar=True)

        return loss
