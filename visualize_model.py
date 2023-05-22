from multimodal.model_swin import MultiModModelSwinEnc
from multimodal_dataset import NCANDADataModule
import nibabel as nib
import numpy as np
import torch
from multimodal_dataset import resize
from settings import IMAGE_SIZE


checkpoint_path = "/home/users/tulikaj/NCANDA_Contrastive/lightning_logs/lxf2jrac/checkpoints/epoch=0-step=70.ckpt"

model = MultiModModelSwinEnc.load_from_checkpoint(checkpoint_path)

# set in evaluation mode (no grads)
model.eval()

# # load val data
# data = NCANDADataModule()
# data.prepare_data()
# #val_loader = data.val_dataloader()
# val_data = data.validation

image_path = "/home/groups/kpohl/ncanda-multi-modal/T1/NCANDA_S00083.nii.gz"
image = nib.load(image_path)
image = image.get_data()
print(type(image))
image = np.array(image, dtype=np.float32)
image = image / image.max()
image = resize(image, (IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE))

image = torch.from_numpy(image)
print("after converting to tensor, image shape is : {}".format(image.shape))

# predict with the model
y_hat = model(image)

#
# # load val data
# data = NCANDADataModule()
# data.prepare_data()
# val_loader = data.val_dataloader()
#
#
# # predict with model
#y_hat = model()