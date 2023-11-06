import torch
from torch import nn
from pytorch_lightning.core.module import LightningModule
from torch.nn import functional as F
from monai.networks.nets import resnet10

class ResNetModel(LightningModule):
    def __init__(self):
        super().__init__()

        self.resnet = resnet10(
            pretrained=False,
            spatial_dims=3,
            n_input_channels=8,
        )

        # add a new fc layer
        self.fc = nn.Linear(400, 5)

        # combine the nets
        self.net = nn.Sequential(self.resnet, self.fc)

    def forward(self, x):
        out = self.net(x)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.to(torch.long)  # Ensure y is of type LongTensor

        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y)

        acc = (torch.argmax(y_pred, dim=1) == y).float().mean()

        # Log loss and accuracy
        self.log('train_loss', loss)
        self.log('train_acc', acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.to(torch.long)  # Ensure y is of type LongTensor

        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y)

        acc = (torch.argmax(y_pred, dim=1) == y).float().mean()

        # Log loss and accuracy
        self.log('val_loss', loss)
        self.log('val_acc', acc, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.to(torch.long)  # Ensure y is of type LongTensor

        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y)

        acc = (torch.argmax(y_pred, dim=1) == y).float().mean()

        # Log loss and accuracy
        self.log('test_loss', loss)
        self.log('test_acc', acc, prog_bar=True)

        return loss
