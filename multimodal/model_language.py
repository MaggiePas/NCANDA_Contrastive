import torch
from torch import nn
from pytorch_lightning.core.module import LightningModule
from torch.nn import functional as F
from monai.networks.nets import resnet10, resnet18, resnet34, resnet50
from settings import IMAGE_SIZE, FEATURES, BATCH_SIZE, TARGET
import torchmetrics
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class MultiModModelWithLanguage(LightningModule):
    '''
    Resnet Model Class including the training, validation and testing steps
    '''

    def __init__(self, class_weight, scaler):

        super().__init__()

        self.class_weight = class_weight

        self.scaler = scaler

        self.resnet = resnet10(pretrained=False,
                               spatial_dims=3,
                               num_classes=120,
                               n_input_channels=1
                               )

        self.tokenizer = AutoTokenizer.from_pretrained('michiyasunaga/BioLinkBERT-base',
                                                       cache_dir="/scratch/users/paschali/")
        self.language_model = AutoModel.from_pretrained('michiyasunaga/BioLinkBERT-base',
                                                        cache_dir="/scratch/users/paschali/")
                                                
        # Freeze weights so those don't get trained        
        for param in self.language_model.parameters():
            param.requires_grad = False


        self.NUM_FEATURES = len(FEATURES)

        # self.NUM_FEATURES = 0

        # fc layer to make image size same as tabular data size
        # self.fc = nn.Linear(400, 1)

        # combine resnet with final fc layer
        # self.imagenet = nn.Sequential(self.resnet, self.fc)
        # fc layer that maps language model inputs to smaller dimension
        self.language_fc = nn.Linear(768, 120)

        # fc layer for tabular data. We substract 2 because age and sex are encoded as sentences
        self.fc1 = nn.Linear((self.NUM_FEATURES - 2), 120)

        # first fc layer which takes concatenated input
        self.fc2 = nn.Linear((120 + 120 + 120), 32)

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

    def forward(self, img, tab):
        """

        x is the input data

        """
        # run the model for the image
        self.language_model = self.language_model.to('cuda')
        # self.tokenizer = self.tokenizer
        
        # print(img.shape)
        img = torch.unsqueeze(img, 1)
        img = img.to(torch.float32)
        # print(img.type)
        # print(img.shape)
        img = self.resnet(img)
        
        batch_sentences = self.get_batch_sentences(tab)
        
        # change the dtype of the tabular data
        tab = tab.to(torch.float32)
        
        ind_to_keep = list(range(0, self.NUM_FEATURES))

        ind_to_keep.remove(2)

        ind_to_keep.remove(3)

        # Remove age and sex from tabular vector since we are using them as language model input
        tab_without_age_sex = tab[:,ind_to_keep]
        
        # forward tabular data
        tab_without_age_sex = F.relu(self.fc1(tab_without_age_sex))

        language_inputs = self.tokenizer(batch_sentences, return_tensors="pt")
        
        language_inputs = language_inputs.to('cuda')

        language_outputs = self.language_model(**language_inputs)

        # 1 x 768
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token. We assume that this has been pre-trained
        pooled_states = language_outputs.pooler_output

        language_features_compressed = self.language_fc(pooled_states)

        # concat image, tabular data and data from language model
        x = torch.cat((img, tab_without_age_sex, language_features_compressed), dim=1)

        x = F.relu(self.fc2(x))

        out = self.fc3(x)

        out = torch.squeeze(out)

        return out

    def get_batch_sentences(self, tabular_to_encode):
        # return_tensors pt means pytorch
        tabular_to_encode = self.scaler.inverse_transform(tabular_to_encode.detach().cpu().numpy())

        batch_age = tabular_to_encode[:, 2]
        batch_sex = tabular_to_encode[:, 1]

        batch_sex_l = list(batch_sex)
        batch_age_l = list(batch_age.round(2))

        batch_sex_l = list(map(lambda x: 'female' if x == 0 else 'male', batch_sex_l))

        batch_pairs = list(zip(batch_sex_l, batch_age_l))

        batch_sentences = ["This subject is " + pair[0] + " and " + str(pair[1]) + " years old" for pair in batch_pairs]

        return batch_sentences

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 35, 50], gamma=0.8)

        lr_scheduler = {
            'scheduler': scheduler,
            'name': 'lr_logging'
        }

        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):

        img, tab, y, subject_id = batch

        # img = torch.tensor(img).float()

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
        self.val_results_df['age'] = tab_bef_normalization[:, 2]
        self.val_results_df['sex'] = tab_bef_normalization[:, 1]

        self.val_results_df_all = pd.concat([self.val_results_df_all, self.val_results_df], ignore_index=True)

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

        loss = loss_func(y_pred,
                         y.squeeze())  # loss = F.binary_cross_entropy(torch.sigmoid(y_pred), y.squeeze(), pos_weights = )

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

        filename_out = '/home/users/paschali/results/train_out_language_' + str(
            self.current_epoch) + '_' + TARGET + '_' + self.trainer.logger.experiment.name + '.csv'

        self.train_results_df_all.to_csv(filename_out)

        # Clear the dataframe so the new epoch can start fresh
        self.train_results_df_all = pd.DataFrame(columns=self.results_column_names)
        # log epoch metric
        self.log('train_acc_epoch', self.train_accuracy)
        self.log('train_macro_acc_epoch', self.train_macro_accuracy)

    def validation_epoch_end(self, outputs):
        # log epoch metric

        filename_out = '/home/users/paschali/results/val_out_language_' + str(
            self.current_epoch) + '_' + TARGET + '_' + self.trainer.logger.experiment.name + '.csv'

        self.val_results_df_all.to_csv(filename_out)

        # Clear the dataframe so the new epoch can start fresh
        self.val_results_df_all = pd.DataFrame(columns=self.results_column_names)

        self.log('val_acc_epoch', self.val_accuracy)
        self.log('val_macro_acc_epoch', self.val_macro_accuracy)

