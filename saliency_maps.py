from multimodal.model_swin import MultiModModelSwinEnc
from multimodal_dataset import NCANDADataModule
import nibabel as nib
import numpy as np
import torch
from medcam import medcam
import os
from multimodal.model import MultiModModel
import matplotlib.pyplot as plt
import matplotlib.cm as cm


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

i = 0
for batch in val_loader:
    i += 1
    slc = saliency(batch[0], batch[1], model)
    print(type(slc), slc.shape)
    # show slices of img and slc
    img = batch[0]
    """
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    print(f"shape of img sliced: {img[0, :, :, 64].detach().numpy().shape}")
    plt.imshow(img[0, :, :, 64].detach().numpy(), cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1, 2, 2)
    plt.imshow(slc[:, :, 64].numpy(), cmap='viridis', vmin=0.4, vmax=1)
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.savefig("saliency_x_y_slice.png")
    """

    # Create a new figure
    fig, ax = plt.subplots()

    # Plot the base image
    ax.imshow(img[0, :, :, 64].detach().numpy(), cmap='gray')

    slc_plot = slc[:, :, 64].numpy()
    #slc_plot[slc_plot < 0.6] = 0
    slc_plot = np.ma.masked_where(slc_plot == 0, slc_plot)
    # Plot the overlay image with adjusted transparency
    my_cmap = cm.plasma
    my_cmap.set_under('k', alpha=0)
    #heatmap = ax.imshow(slc[:, :, 64].numpy(), cmap='viridis')
    heatmap = ax.imshow(slc[:, :, 64].numpy(), cmap=my_cmap, interpolation='hanning', clim=[0.535, 0.75]) # 0.545 to 0.75 for no interpolation

    # Add a colorbar to the plot using the axes
    colorbar = plt.colorbar(heatmap, ax=ax)

    plt.savefig(f"saliency_x_y_slice_{i}.png")

    
