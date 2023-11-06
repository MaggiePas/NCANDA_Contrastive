import torch
from torch import nn
from pytorch_lightning.core.module import LightningModule
from torch.nn import functional as F
from monai.networks.nets import resnet10

class ResNetModel(LightningModule):
    '''
    Resnet Model Class including the training, validation, and testing steps
    '''

    def __init__(self):
        super().__init__()

        self.resnet = resnet10(pretrained=False, spatial_dims=3, n_input_channels=8)

        # add a new fc layer
        self.fc = nn.Linear(400, 5*8)

        # combine the nets
        self.net = nn.Sequential(self.resnet, self.fc)

    def forward(self, x):
        x = torch.unsqueeze(x, 0)
        out = self.net(x)
        out = out.view(-1, 5)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.to(torch.float32)

        y_pred = self(x)
        y_pred = y_pred.to(torch.float32)

        loss = F.cross_entropy(y_pred, y)
        y_pred = torch.argmax(y_pred, dim=1)
        acc = (y_pred == y).float().mean()

        # Log loss and accuracy
        self.log('train_loss', loss)
        self.log('train_acc', acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.to(torch.float32)
        


        y_pred = self(x)
        y_pred = y_pred.to(torch.float32)

        
        loss = F.cross_entropy(y_pred, y)
        y_pred = torch.argmax(y_pred, dim=1)
        acc = (y_pred == y).float().mean()


        # Log loss and accuracy
        self.log('val_loss', loss)
        self.log('val_acc', acc, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.to(torch.float32)

        y_pred = self(x)
        y_pred = y_pred.to(torch.float32)

        loss = F.cross_entropy(y_pred, y)
        y_pred = torch.argmax(y_pred, dim=1)
        acc = (y_pred == y).float().mean()

        # Log loss and accuracy
        self.log('test_loss', loss)
        self.log('test_acc', acc, prog_bar=True)

        return loss
