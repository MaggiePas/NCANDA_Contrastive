import torch
from torch import nn
from pytorch_lightning.core.module import LightningModule
from torch.nn import functional as F
from monai.networks.nets import resnet10, resnet18, resnet34, resnet50


class ResNetModel(LightningModule):
    '''
    Resnet Model Class including the training, validation and testing steps
    '''

    def __init__(self):

        super().__init__()

        self.resnet = resnet10(pretrained=False,
                               spatial_dims=3,
                               n_input_channels=1,
                               )

        # add a new fc layer
        self.fc = nn.Linear(400, 1)

        # combine the nets
        self.net = nn.Sequential(self.resnet, self.fc)

    def forward(self, x):
        """

        x is the input data

        """
        x = torch.unsqueeze(x, 0)

        out = self.net(x)

        out = torch.squeeze(out)

        return out

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

        return optimizer

    def training_step(self, batch, batch_idx):

        x, y = batch
        y = y.to(torch.float32)

        y_pred = self(x)

        loss = F.binary_cross_entropy(torch.sigmoid(y_pred), y.squeeze())

        # Log loss
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):

        x, y = batch
        y = y.to(torch.float32)

        y_pred = self(x)

        loss = F.binary_cross_entropy(torch.sigmoid(y_pred), y.squeeze())

        # Log loss
        self.log('val_loss', loss)

        return loss

    def test_step(self, batch, batch_idx):

        x, y = batch
        y = y.to(torch.float32)

        y_pred = self(x)

        loss = F.binary_cross_entropy(torch.sigmoid(y_pred), y.squeeze())

        self.log("loss", loss)

        return loss
