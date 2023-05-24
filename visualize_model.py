from multimodal.model_swin import MultiModModelSwinEnc
from multimodal_dataset import NCANDADataModule
import nibabel as nib
import numpy as np
import torch
from multimodal_dataset import resize
from settings import IMAGE_SIZE
from medcam import medcam

from multimodal.model import MultiModModel


checkpoint_folder = "2tlih8au"
checkpoint_file = "epoch=9-step=700.ckpt"
checkpoint_path = f'/home/users/tulikaj/NCANDA_Contrastive/lightning_logs/{checkpoint_folder}/checkpoints/{checkpoint_file}'
layer_name = "swin_enc.encoder1.layer.conv1.conv"

model = MultiModModelSwinEnc.load_from_checkpoint(checkpoint_path)

#print(model)

# Inject model with M3d-CAM
model = medcam.inject(model, output_dir=f"attention_maps/{checkpoint_folder}/",
                     backend='ggcam', layer=(layer_name), save_maps=True)
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


"""
image_path = "/home/groups/kpohl/ncanda-multi-modal/T1/NCANDA_S00083.nii.gz"
image = nib.load(image_path)
image = image.get_fdata()
print(type(image))
image = np.array(image, dtype=np.float32)
image = image / image.max()
image = resize(image, (IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE))

image = torch.from_numpy(image)
image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE))
print("after converting to tensor, image shape is : {}".format(image.shape))

# predict with the model
y_hat = model(image, tab=None)
print(y_hat)
"""


