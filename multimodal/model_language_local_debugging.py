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

        self.train_macro_f1 = torchmetrics.classification.MulticlassF1Score(task='multiclass', num_classes=2, average='macro')

        self.train_auc = torchmetrics.classification.BinaryAUROC(task='multiclass', num_classes=2, average='macro')

        self.val_macro_f1 = torchmetrics.classification.MulticlassF1Score(task='multiclass', num_classes=2, average='macro')

        self.val_auc = torchmetrics.classification.BinaryAUROC(task='multiclass', num_classes=2, average='macro')

        self.test_macro_f1 = torchmetrics.classification.MulticlassF1Score(task='multiclass', num_classes=2, average='macro')
        
        self.test_auc = torchmetrics.classification.BinaryAUROC(task='multiclass', num_classes=2, average='macro')

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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #if device.type == "cpu":
        tabular_to_encode = self.scaler.inverse_transform(tabular_to_encode.detach().cpu().numpy())
        # else: 
        #     tabular_to_encode = self.scaler.inverse_transform(tabular_to_encode.detach().cuda().numpy())
        batch_age = tabular_to_encode[:, 2]
        batch_sex = tabular_to_encode[:, 1]
        batch_cahalan_score = tabular_to_encode[:, 8]
        batch_excess_bl_drinking_2 = tabular_to_encode[:, 9]
        batch_lssaga_dsm4_youth_d04_diag = tabular_to_encode[:, 10]
        batch_lssaga_dsm4_youth_d05_diag = tabular_to_encode[:, 11]
        batch_highrisk_yss_extern = tabular_to_encode[:, 12]
        batch_highrisk_yss_intern = tabular_to_encode[:, 13]
        batch_highrisk_pss_extern = tabular_to_encode[:, 14]
        batch_highrisk_pss_intern = tabular_to_encode[:, 15]
        batch_highrisk_youthreport1_yfhi4 = tabular_to_encode[:, 16]
        batch_highrisk_youthreport1_yfhi3 = tabular_to_encode[:, 17]
        batch_highrisk_youthreport1_yfhi5 = tabular_to_encode[:, 18]
        batch_highrisk_youthreport1_leq_c_cnc = tabular_to_encode[:, 19]
        batch_hispanic = tabular_to_encode[:, 3]
        batch_race = tabular_to_encode[:, 4]
        batch_bmi = tabular_to_encode[:, 7]
        batch_rsq_problem_solving = tabular_to_encode[:, 56]
        batch_rsq_emotion_expression = tabular_to_encode[:, 57]
        batch_rsq_acceptance = tabular_to_encode[:, 58]
        batch_rsq_positive_thinking = tabular_to_encode[:, 59]
        batch_rsq_emotion_regulation = tabular_to_encode[:, 60]
        batch_rsq_cognitive_restructuring = tabular_to_encode[:, 61]

        batch_sex_l = list(batch_sex)
        batch_age_l = list(batch_age.round(2))
        batch_cahalan_score_l = list(batch_cahalan_score)
        batch_excess_bl_drinking_2_l = list(batch_excess_bl_drinking_2)
        batch_lssaga_dsm4_youth_d04_diag_l = list(batch_lssaga_dsm4_youth_d04_diag)
        batch_lssaga_dsm4_youth_d05_diag_l = list(batch_lssaga_dsm4_youth_d05_diag)
        batch_highrisk_yss_extern_l = list(batch_highrisk_yss_extern)
        batch_highrisk_yss_intern_l = list(batch_highrisk_yss_intern)
        batch_highrisk_pss_extern_l = list(batch_highrisk_pss_extern)
        batch_highrisk_pss_intern_l = list(batch_highrisk_pss_intern)
        batch_highrisk_youthreport1_yfhi4_l= list(batch_highrisk_youthreport1_yfhi4)
        batch_highrisk_youthreport1_yfhi3_l = list(batch_highrisk_youthreport1_yfhi3)
        batch_highrisk_youthreport1_yfhi5_l = list(batch_highrisk_youthreport1_yfhi5)
        batch_highrisk_youthreport1_leq_c_cnc_l = list(batch_highrisk_youthreport1_leq_c_cnc)


        batch_sex_l = list(map(lambda x: 'girl' if x == 0 else 'boy', batch_sex_l))
        batch_cahalan_score_l = list(map(lambda x: 'nondrinker' if x == 0 else "moderate drinker" if x ==1 else "heavy drinker" if x ==2 else "heavy binge drinker"), batch_cahalan_score_l)
        batch_excess_bl_drinking_2_l = list(map(lambda x: "not a binge drinker" if x ==0 else "a binge drinker" if x ==1 else "a heavy marijuana user but not binker drinker" if x ==2 else "recently became a binge drinker" if x ==3 else "a former binge drinker who has since stopped binge drinking", batch_excess_bl_drinking_2_l))
        batch_lssaga_dsm4_youth_d04_diag_l = list(map(lambda x: "has been diagnosed with alcohol dependence disorder" if x ==1 else "has not been diagnosed with alcohol dependence disorder", batch_lssaga_dsm4_youth_d04_diag_l))
        batch_lssaga_dsm4_youth_d05_diag_l = list(map(lambda x:"has been diagnosed with marijuana dependence disorder" if x ==1 else "has not been diagnosed with marijuana dependence disorder", batch_lssaga_dsm4_youth_d05_diag_l))
        batch_highrisk_youthreport1_yfhi4_l = list(map(lambda x: "does not have a blood relative who has had problems with drugs" if x == 0 else "has a blood relative who has had problems with drugs", batch_highrisk_youthreport1_yfhi4_l))
        batch_highrisk_youthreport1_yfhi3_l = list(map(lambda x: "does not have a blood relative who has had problems with alcohol" if x == 0 else "has a blood relative who has had problems with alcohol", batch_highrisk_youthreport1_yfhi3_l))
        batch_highrisk_youthreport1_yfhi5_l = list(map(lambda x: "does not have a blood relative who saw visions or heard voices or thought people were spying on them or plotting against them" if x == 0 else "has a blood relative who saw visions or heard voices or thought people were spying on them or plotting against them", batch_highrisk_youthreport1_yfhi5_l))

        batch_pronoun_l = list(map(lambda x: 'She' if x == 0 else 'He', batch_sex_l))

        batch_hispanic_l = list(batch_hispanic)
        batch_hispanic_l = list(map(lambda x: 'not Hispanic' if x == 0 else 'Hispanic', batch_hispanic_l))

        batch_race_l = list(batch_race)
        batch_race_l = list(map(lambda x: 'Native American or American Indian' if x == 1 else
                                ("Asian" if x == 2 else
                                 ("Pacific Islander" if x == 3 else
                                  ("African-American or Black" if x == 4 else
                                   ("Caucasian or White" if x == 5 else
                                    ("Other race or race not specified"))))), batch_race_l))
        
        batch_bmi_l = list(batch_bmi)
        batch_bmi_l = list(batch_bmi_l.round(2))

        batch_rsq_problem_solving_l = list(batch_rsq_problem_solving)
        rsq_problem_solving_list = list("\"I try to think of different ways to change the problem to fix the situation,\" ",
                                     "\"I ask other people for help or ideas about how to make the problem better,\" ",
                                     "and " + "\"I do something to try to fix the problem or take action to change things.\" ")
        batch_rsq_problem_solving_l = list(map(lambda x: 'mostly disagrees with sentiments like ' + rsq_problem_solving_list if  1 < x <= 2 
                                               else ('sometimes agrees and sometimes disagrees with sentiments like ' + rsq_problem_solving_list if  2 < x <= 3
                                               else 'mostly agrees with sentiments like '), rsq_problem_solving_list))
        
        batch_rsq_emotion_expression_l = list(batch_rsq_emotion_expression)
        rsq_emotion_expression_list = list("\"I let someone or something know how I feel,\" ",
                                     "\"I get sympathy, understanding, or support from someone (like a parent, friend, brother/sister, or teacher),\" ",
                                     "and " + "\"I let my feelings out (like by writing in my journal/diary, drawing/painting, complaining to let off steam, being sarcastic/making fun, listening to music, exercising, yelling, crying).\" ")
        batch_rsq_emotion_expression_l = list(map(lambda x: 'mostly disagrees with sentiments like ' + rsq_emotion_expression_list if  1 < x <= 2 
                                               else ('sometimes agrees and sometimes disagrees with sentiments like ' + rsq_emotion_expression_list if  2 < x <= 3
                                               else 'mostly agrees with sentiments like '), rsq_emotion_expression_list))
        
        batch_batch_rsq_acceptance_l = list(batch_rsq_acceptance)
        rsq_acceptance_list = list("\"I decide I'm okay with the way I am, even though I'm not perfect,\" ",
                                     "\"I realize that I have to live with things the way they are,\" ",
                                     "and " + "\"I just take things as they are; I go with the flow.\" ")
        batch_batch_rsq_acceptance_l = list(map(lambda x: 'mostly disagrees with sentiments like ' + rsq_acceptance_list if  1 < x <= 2 
                                               else ('sometimes agrees and sometimes disagrees with sentiments like ' + rsq_acceptance_list if  2 < x <= 3
                                               else 'mostly agrees with sentiments like '), rsq_acceptance_list))
        
        batch_rsq_positive_thinking_l = list(batch_rsq_positive_thinking)
        rsq_positive_thinking_list = list("\"I tell myself that I can get through this, or that I'll do better next time,\" ",
                                     "\"I tell myself that everything will be alright,\" ",
                                     "and " + "\"I think of ways to laugh about it so that it won't seem so bad.\" ")
        batch_rsq_positive_thinking_l= list(map(lambda x: 'mostly disagrees with sentiments like ' + rsq_positive_thinking_list if  1 < x <= 2 
                                               else ('sometimes agrees and sometimes disagrees with sentiments like ' + rsq_positive_thinking_list if  2 < x <= 3
                                               else 'mostly agrees with sentiments like '), rsq_positive_thinking_list))
        
        batch_rsq_emotion_regulation_l = list(batch_rsq_emotion_regulation)
        rsq_emotion_regulation_list = list("\"I get help from others when I'm trying to figure out how to deal with my feelings,\" ",
                                     "\"I do something to calm myself down when I'm having problems with others (like take deep breaths, listen to music, pray, take a break, walk, meditate),\" ",
                                     "and " + "\"I keep my feelings under control when I have to, then let them out when they won't make things worse.\" ")
        batch_rsq_emotion_regulation_l= list(map(lambda x: 'mostly disagrees with sentiments like ' + rsq_emotion_regulation_list if  1 < x <= 2 
                                               else ('sometimes agrees and sometimes disagree with sentiments like ' + rsq_emotion_regulation_list if  2 < x <= 3
                                               else 'mostly agrees with sentiments like '), rsq_emotion_regulation_list))
        
        batch_rsq_cognitive_restructuring_l = list(batch_rsq_cognitive_restructuring)
        rsq_cognitive_restructuring_list = list("\"I tell myself that things could be worse,\" ",
                                     "\"I tell myself that it doesn't matter, that it isn't a big deal,\" ",
                                     "and " + "\"I think about the things I'm learning from the situation, or something good that will come from it.\" ")
        batch_rsq_cognitive_restructuring_l= list(map(lambda x: 'mostly disagrees with sentiments like ' + rsq_cognitive_restructuring_list if  1 < x <= 2 
                                               else ('sometimes agrees and sometimes disagrees with sentiments like ' + rsq_cognitive_restructuring_list if  2 < x <= 3
                                               else 'mostly agrees with sentiments like '), rsq_cognitive_restructuring_list))

        
        # add description of patients
        batch_pairs = list(zip(batch_sex_l, batch_age_l, batch_cahalan_score_l, batch_excess_bl_drinking_2_l, batch_lssaga_dsm4_youth_d04_diag_l, 
                               batch_lssaga_dsm4_youth_d05_diag_l, batch_highrisk_youthreport1_yfhi4_l, batch_highrisk_youthreport1_yfhi3_l,
                               batch_highrisk_youthreport1_yfhi5_l, batch_hispanic_l, batch_race_l, batch_bmi_l, batch_pronoun_l,
                               batch_rsq_problem_solving_l, batch_rsq_emotion_expression_l, batch_batch_rsq_acceptance_l, batch_rsq_positive_thinking_l,
                               batch_rsq_emotion_regulation_l, batch_rsq_cognitive_restructuring_l))

        batch_sentences = ["This subject is a " + str(pair[1]) + " year old" + str(pair[0]) + " who is a " + str(pair[2]) + " and " + str(pair[3])+ "." +
                           str(pair[12]) + " is " + str(pair[9]) + ", and " + str(pair[10]) + " with a BMI of " + str(pair[11]) + "." +
                           str(pair[12]) + str(pair[4]) + " and " + str(pair[5]) + " and " + str(pair[6]) + " and " + str(pair[7]) + " and " + str(pair[8]) + "."  +
                           str(pair[12]) + str(pair[13]) + str(pair[14]) + str(pair[15]) + str(pair[16]) + str(pair[17]) + str(pair[18]) for pair in batch_pairs]
        print(batch_sentences)
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
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #if device.type == "cpu":
        self.train_results_df['label'] = y.squeeze().detach().cpu().numpy()
        self.train_results_df['prediction'] = y_pred_tag.detach().cpu().numpy()
        tab_bef_normalization = self.scaler.inverse_transform(tab.detach().cpu().numpy())
        # else: 
        #     self.train_results_df['label'] = y.squeeze().detach().cuda().numpy()
        #     self.train_results_df['prediction'] = y_pred_tag.detach().cuda().numpy()
        #     tab_bef_normalization = self.scaler.inverse_transform(tab.detach().cuda().numpy())
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
        filename_out = '/scratch/users/ewesel/train_out_language_'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cpu":
            filename_out = '/Users/emilywesel/Desktop/NCANDA/train_out_language_' 
        filename_out += str(self.current_epoch) + '_' + TARGET + '_' + self.trainer.logger.experiment.name + '.csv'

        self.train_results_df_all.to_csv(filename_out)

        # Clear the dataframe so the new epoch can start fresh
        self.train_results_df_all = pd.DataFrame(columns=self.results_column_names)
        # log epoch metric
        self.log('train_acc_epoch', self.train_accuracy)
        self.log('train_macro_acc_epoch', self.train_macro_accuracy)
        #self.log('train_f1', self.train_macro_f1)
        #self.log('train_auc', self.train_auc)

    def validation_epoch_end(self, outputs):
        # log epoch metric
        filename_out = '/scratch/users/ewesel/val_out_language_'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cpu":
            filename_out = '/Users/emilywesel/Desktop/NCANDA/val_out_language_' 
        filename_out += str(self.current_epoch) + '_' + TARGET + '_' + self.trainer.logger.experiment.name + '.csv'

        self.val_results_df_all.to_csv(filename_out)

        # Clear the dataframe so the new epoch can start fresh
        self.val_results_df_all = pd.DataFrame(columns=self.results_column_names)

        self.log('val_acc_epoch', self.val_accuracy)
        self.log('val_macro_acc_epoch', self.val_macro_accuracy)
        #self.log('val_f1', self.val_macro_f1)
        #self.log('val_auc', self.val_auc)