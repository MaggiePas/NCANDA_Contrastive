from torch.utils.data import Dataset
import pandas as pd
import torch
import os
import nibabel as nib
import torchio as tio
import pytorch_lightning as pl
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")


from settings import CSV_FILE, IMAGE_PATH, IMAGE_SIZE, VAL_SIZE, TEST_SIZE, FEATURES, TARGET, BATCH_SIZE, transformation, target_transformations
IMAGE_PATH = r'/scratch/users/ewesel/data/chest_scans'
CSV_FILE = r'/scratch/users/ewesel/data/scores.csv'
IMAGE_SIZE = 128
IMAGE_SIZE0 = 53
TARGET = 'total_bin'
from torch.utils.data import DataLoader
from scipy.interpolate import interpn
from sklearn.preprocessing import MinMaxScaler

def resize(mat, new_size, interp_mode='linear'):
    """
    resize: resamples a "matrix" of spatial samples to a desired "resolution" or spatial sampling frequency via interpolation
    Args:        mat:                matrix to be "resized" i.e. resampled
        new_size:         desired output resolution
        interp_mode:        interpolation method
    Returns:
        res_mat:            "resized" matrix
    """
    
    mat = mat.squeeze()
    mat_shape = mat.shape

    axis = []
    for dim in range(len(mat.shape)):
        dim_size = mat.shape[dim]
        axis.append(np.linspace(0, 1, dim_size))

    new_axis = []
    for dim in range(len(new_size)):
        dim_size = new_size[dim]
        new_axis.append(np.linspace(0, 1, dim_size))

    points = tuple(p for p in axis)
    xi = np.meshgrid(*new_axis)
    xi = np.array([x.flatten() for x in xi]).T
    new_points = xi
    mat_rs = np.squeeze(interpn(points, mat, new_points, method=interp_mode))
    # TODO: fix this hack.
    if dim + 1 == 3:
        mat_rs = mat_rs.reshape([new_size[1], new_size[0], new_size[2]])
        mat_rs = np.transpose(mat_rs, (1, 0, 2))
    else:
        mat_rs = mat_rs.reshape(new_size, order='F')
    # update command line status
    assert mat_rs.shape == tuple(new_size), "Resized matrix does not match requested size."
    return mat_rs
    
def categorize_total(total):
    if total == 0:
        return 1
    elif total <= 10:
        return 2
    elif total <= 100:
        return 3
    elif total <= 400:
        return 4
    else:
        return 5
    
class ASDataset(Dataset):
    
    def __init__(self, image_dir, subjects, transform, target_transform):
        self.image_dir = image_dir
        self.transform = transform
        self.target_transform = target_transform
        self.subjects = subjects
        # self.input_tab = input_tabular
        # df = pd.read_csv(CSV_FILE)
        # self.X = list(df["filename"])
        # self.X_train = df[df['filename'].isin(subjects)]
        # df['total_bin'] = df['total'].apply(categorize_total)
        # labels = list(df['total_bin'])
        # labels.insert(0, 1)
        # labels.insert(50, labels[-1])
        # self.y = labels

    def __len__(self):
        print(len(self.X))
        return len(self.X)

    def __getitem__(self, idx):
        # Convert idx from tensor to list due to pandas bug (that arises when using pytorch's random_split)
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        # print(f'{self.csv_df_split.iloc[idx, 0]}\n')
        image_name = os.path.join(self.image_dir, 'M0_')#self.input_tab.iloc[idx, 0])

        subject_id = self.input_tab.iloc[idx, 0]

        image_path = image_name + '.nii.gz'

        image = nib.load(image_path)
        image = image.get_fdata()

        # change to numpy
        image = np.array(image, dtype=np.float32)

        # scale images between [0,1]
        image = image[0:53, 100:350, 175:425]

        image = image / image.max()

        image = resize(image, (IMAGE_SIZE0, IMAGE_SIZE, IMAGE_SIZE))

        if (self.transform and np.random.choice([0, 1]) == 0 ):
            transform = tio.RandomAffine(
            scales=(0.9, 1.2),
            degrees=10,
            )
            image = torch.tensor(image)

            # Add a singleton channel dimension to convert it to 4D
            image = torch.unsqueeze(image, 0)

            # Convert the Torch tensor to a TorchIO ScalarImage
            tio_image = tio.ScalarImage(tensor=image)

            # Apply the transformation to the image
            transformed_image = transform(tio_image)

            # Access the transformed image as a Torch tensor
            image = transformed_image.data.squeeze(0)

            # Convert the Torch tensor back to a NumPy array
            image = image.numpy()
            
            # temp = img[0]
            # image_data_new = temp.reshape(1, 64, 64, 64)
            # transformed = transform(image_data_new)
            # img = transformed

        
        label = self.y.values[idx]
        # tab = self.X.values[idx]

        if self.target_transform:
            label = self.target_transform(label)

        # return image, tab, label, subject_id
        return image, label, subject_id


class ASDataModule(pl.LightningDataModule):

    def __init__(self):
        super().__init__()

    def get_stratified_split(self, csv_file):
        group_by_construct_train = {1: [], 0: []}
        group_by_construct_test = {1: [], 0: []}
        df = pd.read_csv(csv_file)
        X = list(df["filename"])
        print("X", X)

        df['total_bin'] = df['total'].apply(categorize_total)
        labels = list(df['total_bin'])
        # labels.insert(0, 1)
        # labels.insert(50, labels[-1])
        all_labels = labels
        train_subj, test_subj, y_train, y_test = train_test_split(X, all_labels, stratify=all_labels)
        print("train_subj", train_subj,"\n train_sub", test_subj, "\n y_train:", y_train)
        train_subj_df = df[df['filename'].isin(list(train_subj))]

        test_subj_df = df[df['filename'].isin(list(test_subj))]

        for subject in train_subj:
            subj_visits = df[df['filename'] == subject]
            print((int)(subject), len(labels))
            subj_label = labels[(int)(subject)]
            # group_by_construct_train[subj_label].append(subject)

        for subject in test_subj:
            subj_visits = df[df['filename'] == subject]
            print((int)(subject), len(labels))
            subj_label = labels[(int)(subject)]
            # group_by_construct_test[subj_label].append(subject)

        return train_subj, test_subj, y_train, y_test#, group_by_construct_train, group_by_construct_test

  
        
    def calculate_class_weight(self, X_train):

        y_train = X_train.loc[:, TARGET]
        number_neg_samples = np.sum(y_train.values == False)
        num_pos_samples = np.sum(y_train.values == True)
        mfb = number_neg_samples / num_pos_samples

        self.class_weights = mfb

        return mfb

    def prepare_data(self):

        train_subj, test_subj, y_train, y_test = self.get_stratified_split(CSV_FILE)
                
        # self.class_weight = self.calculate_class_weight(X_train)

        self.train = ASDataset(image_dir=IMAGE_PATH, subjects = train_subj, transform=transformation,
                                   target_transform=target_transformations)

        print(f'Train dataset length: {self.train.__len__()}')

        self.validation = ASDataset(image_dir=IMAGE_PATH, subjects = test_subj, transform=transformation,
                                        target_transform=target_transformations)
                                        
        print(f'Validation dataset length: {self.validation.__len__()}')
        self.test = self.validation

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, drop_last = False)

    def val_dataloader(self):
        return DataLoader(self.validation, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, drop_last = True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, drop_last = True)

