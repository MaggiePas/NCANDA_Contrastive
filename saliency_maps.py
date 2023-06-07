from multimodal.model_swin import MultiModModelSwinEnc
from multimodal_dataset import NCANDADataModule
import nibabel as nib
import numpy as np
import torch
from medcam import medcam
import os
from multimodal.model import MultiModModel
import matplotlib.pyplot as plt


#checkpoint_folder = "swin_UNETR_12_cross_corr_128ImgSize"
checkpoint_folder  = "e48qyy41"
checkpoint_file = "epoch=0-step=424.ckpt"
checkpoint_path = f'/home/users/tulikaj/NCANDA_Contrastive/lightning_logs/{checkpoint_folder}/checkpoints/{checkpoint_file}'
layer_name = "swin_enc"
output_dir = f"attention_maps/{checkpoint_folder}/"
store_png_here = "final_pngs"


def saliency(img,tab, model):
    for param in model.parameters():
        param.requires_grad = False
    # set in evaluation mode (no grads)
    model.eval()
    # set input grad to true
    img.requires_grad = True
    pred = model(img,tab)
    score = pred
    # backward pass to get gradients of score predicted class w.r.t. input image
    score.backward()
    # get max along channel axis
    print(f"shape of img grad:{img.grad.shape}")
    slc = img.grad[0]
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
    slc = saliency(batch[0], batch[1], model)
    print(type(slc), slc.shape)
    break
