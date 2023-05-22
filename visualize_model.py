from multimodal.model_swin import MultiModModelSwinEnc
from multimodal_dataset import NCANDADataModule
import nibabel as nib
import numpy as np
import torch
from multimodal_dataset import resize
from settings import IMAGE_SIZE
from medcam import medcam


checkpoint_path = "/home/users/tulikaj/NCANDA_Contrastive/lightning_logs/lxf2jrac/checkpoints/epoch=0-step=70.ckpt"

model = MultiModModelSwinEnc.load_from_checkpoint(checkpoint_path)

# Inject model with M3d-CAM
model = medcam.inject(model, output_dir="attention_maps", save_maps=True)

# set in evaluation mode (no grads)
model.eval()

data = NCANDADataModule()
data.prepare_data()
val_loader = data.val_dataloader()

for batch in val_loader:
    print(type(batch))
    print(len(batch))
    # Every time forward is called, attention maps will be generated and saved in the directory "attention_maps"
    output = model(batch)


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


