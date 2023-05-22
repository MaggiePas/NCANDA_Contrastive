from multimodal.model_swin import MultiModModelSwinEnc
from multimodal_dataset import NCANDADataModule


checkpoint_path = "/home/users/tulikaj/NCANDA_Contrastive/lightning_logs/lxf2jrac/checkpoints/epoch=0-step=70.ckpt"

model = MultiModModelSwinEnc.load_from_checkpoint(checkpoint_path)

# set in evaluation mode (no grads)
model.eval()

print(model.scaler)

# load val data
data = NCANDADataModule()
data.prepare_data()
#val_loader = data.val_dataloader()
val_data = data.validation

# predict with the model
y_hat = model(val_data)

#
# # load val data
# data = NCANDADataModule()
# data.prepare_data()
# val_loader = data.val_dataloader()
#
#
# # predict with model
#y_hat = model()