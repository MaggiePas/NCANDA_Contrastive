from multimodal.model_swin import MultiModModelSwinEnc
from multimodal_dataset import NCANDADataModule
import nibabel as nib
import numpy as np
import torch
from medcam import medcam
import os
from multimodal.model import MultiModModel
import matplotlib.pyplot as plt


checkpoint_folder = "swin_UNETR_12_cross_corr_128ImgSize/checkpoints"
checkpoint_file = "epoch=19-step=8480.ckpt"
checkpoint_path = f'/home/users/tulikaj/NCANDA_Contrastive/lightning_logs/{checkpoint_folder}/checkpoints/{checkpoint_file}'
layer_name = "swin_enc"
output_dir = f"attention_maps/{checkpoint_folder}/"
store_png_here = "final_pngs"


def saliency(img, model):
    for param in model.parameters():
        param.requires_grad = False
    # set in evaluation mode (no grads)
    model.eval()
    # set input grad to true
    img.requires_grad = True
    preds = model(img)
    score, indices = torch.max(preds, 1)
    # backward pass to get gradients of score predicted class w.r.t. input image
    score.backward()
    # get max along channel axis
    slc, _ = img.grad[1]
    # normalize to [0..1]
    slc = (slc - slc.min()) / (slc.max() - slc.min())
    # plot image and its saliency map in
    return slc


print("loading model from checkpoint")
model = MultiModModelSwinEnc.load_from_checkpoint(checkpoint_path)
print("Done")

# load val data
data = NCANDADataModule()
data.prepare_data()
val_loader = data.val_dataloader()

for batch in val_loader:
    slc = saliency(batch[0], model)
    print(type(slc), slc.shape)
    break
