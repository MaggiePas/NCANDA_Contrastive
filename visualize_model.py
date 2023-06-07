from multimodal.model_swin import MultiModModelSwinEnc
from multimodal_dataset import NCANDADataModule
import nibabel as nib
import numpy as np
import torch
from multimodal_dataset import resize
from settings import IMAGE_SIZE
from medcam import medcam
import os
from multimodal.model import MultiModModel
import matplotlib.pyplot as plt


checkpoint_folder = "full_swin_48_pretrained_new"
checkpoint_file = "epoch=19-step=4240.ckpt"
checkpoint_path = f'/home/users/tulikaj/NCANDA_Contrastive/lightning_logs/{checkpoint_folder}/checkpoints/{checkpoint_file}'
layer_name = "swin_enc"
output_dir = f"attention_maps/{checkpoint_folder}/"
store_png_here = "final_pngs"

model = MultiModModelSwinEnc.load_from_checkpoint(checkpoint_path)

#print(model)

# Inject model with M3d-CAM
model = medcam.inject(model, output_dir=output_dir,
                     backend='gcam', layer=(layer_name), save_maps=True)
# set in evaluation mode (no grads)
model.eval()

data = NCANDADataModule()
data.prepare_data()
val_loader = data.val_dataloader()

for batch in val_loader:
    #print(type(batch))
    #print(len(batch))
    #print(batch[0])
    #print(batch[0].shape)
    # Every time forward is called, attention maps will be generated and saved in the directory "attention_maps"
    output = model(batch[0])
    break


count = 0
output_dir += "swin_enc/"

for file in os.listdir(output_dir):
    f = os.path.join(output_dir, file)
    test_load = nib.load(f).get_fdata()
    print(test_load.shape)
    
    test_xy_slice = test_load[:, :, 32]
    plt.imshow(test_xy_slice)
    if count == 0:
        plt.colorbar()
    plt.savefig(f'{store_png_here}/{file}_slice_xy.png')

    test_yz_slice = test_load[32, :, :]
    plt.imshow(test_yz_slice)
    plt.savefig(f'{store_png_here}/{file}_slice_yz.png')

    test_xz_slice = test_load[:, 32, :]
    plt.imshow(test_xz_slice)
    plt.savefig(f'{store_png_here}/{file}_slice_xz.png')

    count += 1




