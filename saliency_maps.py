from multimodal.model_swin import MultiModModelSwinEnc
from multimodal_dataset import NCANDADataModule
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


checkpoint_folder  = "o33cyjwt"
checkpoint_file = "epoch=12-step=5512.ckpt"
checkpoint_path = f'/home/users/tulikaj/NCANDA_Contrastive/lightning_logs/{checkpoint_folder}/checkpoints/{checkpoint_file}'
# store saliency maps here
store_png_here = "final_saliency_maps_temp"


def saliency(img, tab, model):
    for param in model.parameters():
        param.requires_grad = False
    # set in evaluation mode (no grads)
    model.eval()
    # set input grad to true
    img.requires_grad = True
    score = model(img, tab)
    # backward pass to get gradients of score predicted class w.r.t. input image
    score.backward()
    # get max along channel axis
    print(f"shape of img grad:{img.grad.shape}")
    slc = img.grad[0]
    # normalize to [0..1]
    slc = (slc - slc.amin()) / (slc.amax() - slc.amin())
    # plot image and its saliency map in
    return slc


print("loading model from checkpoint")
model = MultiModModelSwinEnc.load_from_checkpoint(checkpoint_path)
print("Done")

# load val data
data = NCANDADataModule()
data.prepare_data()
train_loader = data.train_dataloader(shuffle=False)


def save_plots_slice(img, slc, plane, sample_name):
    fig, ax = plt.subplots()
    my_cmap = cm.plasma
    my_cmap.set_under('k', alpha=0)
    # plot base image of the brain
    if plane == 'xy':
        img_plot = np.rot90(img[0, :, :, 64].detach().numpy(), k=1)
        slc_plot = np.rot90(slc[:, :, 64].numpy(), k=1)
    elif plane == 'yz':
        img_plot = np.fliplr(np.rot90(img[0, 64, :, :].detach().numpy(), k=1))
        slc_plot = np.fliplr(np.rot90(slc[64, :, :].numpy(), k=1))
    else:
        img_plot = np.rot90(img[0, :, 64, :].detach().numpy(), k=1)
        slc_plot = np.rot90(slc[:, 64, :].numpy(), k=1)

    ax.imshow(img_plot, cmap='gray')
    slc_plot = np.ma.masked_where(slc_plot == 0, slc_plot)
    heatmap = ax.imshow(slc_plot, cmap=my_cmap, interpolation='hanning',
                        clim=[0.4, 0.9])  # 0.545 to 0.75 for no interpolation, 0.535 for hanning interpolation
    colorbar = plt.colorbar(heatmap, ax=ax)
    print(f"Saving '{plane}' saliency map slice for sample {sample_name}")
    plt.savefig(f"{store_png_here}/saliency_{plane}_slice_Sample_{sample_name}.png")
    return


i = 0
for batch in train_loader:
    i += 1
    slc = saliency(batch[0], batch[1], model)
    print(type(slc), slc.shape)
    # show slices of img and slc
    img = batch[0]
    save_plots_slice(img, slc, 'xy', batch[3])
    save_plots_slice(img, slc, 'yz', batch[3])
    save_plots_slice(img, slc, 'xz', batch[3])
    #break

    
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
