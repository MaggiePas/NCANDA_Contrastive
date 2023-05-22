from multimodal.model_swin import MultiModModelSwinEnc
from multimodal_dataset import NCANDADataModule


checkpoint_path = "/home/users/tulikaj/NCANDA_Contrastive/lightning_logs/vv51617w/checkpoints/epoch=19-step=1400.ckpt"


model = MultiModModelSwinEnc.load_from_checkpoint(checkpoint_path)

# set in evaluation mode (no grads)
model.eval()

print(model.learning_rate)

#
# # load val data
# data = NCANDADataModule()
# data.prepare_data()
# val_loader = data.val_dataloader()
#
#
# # predict with model
#y_hat = model()