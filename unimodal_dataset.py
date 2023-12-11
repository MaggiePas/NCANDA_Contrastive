from torch.utils.data import Dataset
import pandas as pd
import torch
import os
import nibabel as nib
import pytorch_lightning as pl
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
# from totalsegmentator.python_api import totalsegmentator
# import subprocess
# from collections import Counter
# from torchvision import transforms as v2
# from PIL import Image
# import torchvision.transforms.functional as F
import numpy as np
from scipy.ndimage import rotate

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")


from settings import CSV_FILE, IMAGE_PATH, IMAGE_SIZE, VAL_SIZE, TEST_SIZE, FEATURES, TARGET, BATCH_SIZE, transformation, target_transformations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device.type)
if device.type == 'cpu':
    IMAGE_PATH = r'/Users/emilywesel/Desktop/Heart/chest_scans'
    SEG_PATH = r'/Users/emilywesel/Desktop/Heart/cropped'
    # image_path = '/scratch/users/ewesel/data/chest_scans/M0_5.nii.gz'
    CSV_FILE = r'/Users/emilywesel/Desktop/Heart/scores.csv'
else:
    IMAGE_PATH = r'/scratch/users/ewesel/data/chest_scans'
    SEG_PATH = r'/scratch/users/ewesel/data/cropped'
    CSV_FILE = r'/scratch/users/ewesel/data/scores.csv'
    IMAGE_PATH = SEG_PATH
    # CSV_FILE = r'/scratch/users/ewesel/data/scores_small.csv'


IMAGE_SIZE = 128
IMAGE_SIZE0 = 53
TARGET = 'total_bin'
from torch.utils.data import DataLoader
from scipy.interpolate import interpn
# from sklearn.preprocessing import MinMaxScaler

# from torchvision import transforms

# class RandomRotationWithAngle(transforms.RandomRotation):
#     def __init__(self, degrees, expand=False, center=None):
#         super().__init__(degrees=degrees, expand=expand, center=center)

#     def forward(self, img):
#         angle = self.get_params(self.degrees)
#         return super().forward(img, angle)



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
    # mat_shape = mat.shape

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
# def transform(image, angle):
#     """
#     Apply random rotation to the input image.
#     """
#     # Convert numpy array to PIL image
#     pil_image = Image.fromarray(image.squeeze())
#     # Apply random rotation
#     rotated_image = pil_image.rotate(angle)
#     # Convert back to numpy array
#     rotated_image = np.array(rotated_image, dtype=np.float32)
#     return rotated_image
    
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
    
    def __init__(self, image_dir, subjects, transform, target_transform, labels, rotation_angle=10, train_mode = True):
        self.image_dir = image_dir
        self.transform = transform
        self.target_transform = target_transform
        self.subjects = subjects
        self.labels = labels
        self.rotation_angle = rotation_angle
        self.train_mode = train_mode
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
        return len(self.subjects)
    def get_class_weight(self):
        return self.class_weight

    def __getitem__(self, idx):
        # Convert idx from tensor to list due to pandas bug (that arises when using pytorch's random_split)
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        # print("not idf", idx)
        # print("subjects:", self.subjects)
        subject_id = self.subjects[idx]


        # print(f'{self.csv_df_split.iloc[idx, 0]}\n')
        temp = "heart_cropped"
        # temp = "M0_"
        image_name = os.path.join(self.image_dir, temp+str(subject_id))#self.input_tab.iloc[idx, 0])
        # image_name = os.path.join(self.image_dir, 'heart_cropped'+str(subject_id))#self.input_tab.iloc[idx, 0])

        image_path = image_name + '.nii.gz'
        
        # outputfile = "/scratch/users/ewesel/data/chest_scans/segmentations" + "/"+f'M0_{str(subject_id)}_heart.nii.gz'

        # Run TotalSegmentator command
        # command = 'pip install TotalSegmentator'
        # result = subprocess.run(command, shell=True, capture_output=True, text=True)
        # command = f'TotalSegmentator -i {image_path} -o {outputfile} --roi_subset heart'
        # result = subprocess.run(command, shell=True, capture_output=True, text=True)
        # if result.returncode != 0:
        #     raise RuntimeError(f"TotalSegmentator command failed with error: {result.stderr}")

        # print(result.stdout)
        # mp.set_start_method('spawn', force=True)
        # totalsegmentator(image_path, outputfile, roi_subset= ["heart"])
        # mp.set_start_method('fork', force=True)


        image = nib.load(image_path)
        image = image.get_fdata()


        # change to numpy
        image = np.array(image, dtype=np.float32)

        # scale images between [0,1]
        # image = image[0:53, 100:350, 175:425]

        image = image / image.max()

        image = resize(image, (IMAGE_SIZE0, IMAGE_SIZE, IMAGE_SIZE))
        
        label = self.labels[idx]
        # tab = self.X.values[idx]

        # if self.transform and np.random.rand() < 0.5 and self.train_mode:  # 50% chance of applying rotation

        #     # Assuming you have an image represented as a NumPy array called 'image'
        #     # and you want to rotate it by 45 degrees clockwise
        #     rotation_angle = np.random.uniform(-20, 20)
        #     print("tator tot", rotation_angle)

        #     # Rotate the image
        #     image = rotate(image, rotation_angle, reshape=False)
            # rotater = v2.RandomRotation(degrees=(-10, 10))
            # image = rotater(image)
            # angle = np.random.uniform(-self.rotation_angle, self.rotation_angle)
            # rotated_image = transform(image, angle)
            
            # Example of usage
            # rotation_transform = RandomRotationWithAngle(degrees=(-self.rotation_angle, self.rotation_angle))
            # rotated_image = rotation_transform(image)
            # image = rotated_image

        if self.target_transform:
            label = self.target_transform(label)
        image = image.astype(np.float32)

        if device == 'cpu':
            image = np.ascontiguousarray(image)
            label = np.array(label, dtype=np.float32)


        # return image, tab, label, subject_id
        return image, label#, subject_id


class ASDataModule(pl.LightningDataModule):

    def __init__(self):
        super().__init__()

    def get_stratified_split(self, csv_file):
        # group_by_construct_train = {1: [], 0: []}
        # group_by_construct_test = {1: [], 0: []}
        df = pd.read_csv(csv_file)
        X = list(df["filename"])

        # print("X", X)
        

        df['total_bin'] = df['total'].apply(categorize_total)
        labels = list(df['total_bin'])
        # Use Counter to get counts
        # counts = Counter(labels)

        # Print counts
        # for value, count in counts.items():
            # print(f"emil yyyy {value}: {count} times")


        all_labels = labels
        # print(len(X))
        if len(X) == 1:
            # If there's only one sample, use it for training without splitting
            train_subj, y_train = X, labels
            test_subj, y_test = X, labels
        else: 
            train_subj, test_subj, y_train, y_test = train_test_split(X, all_labels, stratify=all_labels)
        # train_subj_df = df[df['filename'].isin(list(train_subj))]

        # test_subj_df = df[df['filename'].isin(list(test_subj))]
        print(len(train_subj), len(test_subj), len(y_train), len(y_test))


        # for subject in train_subj:
        #     subj_visits = df[df['filename'] == subject]
        #     idx = (int)(subject)
        #     idx -= 1
        #     if idx >= 50:
        #         idx -=1
        #     subj_label = labels[idx]
        #     # group_by_construct_train[subj_label].append(subject)

        # for subject in test_subj:
        #     subj_visits = df[df['filename'] == subject]
        #     idx = (int)(subject)
        #     idx -= 1
        #     if idx >= 50:
        #         idx -=1
        #     subj_label = labels[idx]
        #     # group_by_construct_test[subj_label].append(subject)

        return train_subj, test_subj, y_train, y_test#, group_by_construct_train, group_by_construct_test

        
    def calculate_class_weight(self, labels):

        labels = [2 if x in range(4, 6) else 1 for x in labels]


        
        # Assuming the classes are integers (1, 2, 3, 4, 5)
        unique_classes = np.unique(labels)
        
        class_weights = {}

        for class_label in unique_classes:
            num_samples = np.sum(labels == class_label)
            print(class_label, "has weight", num_samples)
            class_weights[class_label] = num_samples

        # Calculate the total number of samples
        total_samples = len(labels)

        # Calculate class weights
        for class_label, num_samples in class_weights.items():
            class_weights[class_label] = total_samples / (len(unique_classes) * num_samples)

        print(class_weights)
        self.class_weights = class_weights

        return list(class_weights.values())

    def prepare_data(self):

        train_subj, test_subj, y_train, y_test = self.get_stratified_split(CSV_FILE)
                
        self.class_weight = self.calculate_class_weight(y_train)

        self.train = ASDataset(image_dir=IMAGE_PATH, subjects = train_subj, transform=transformation,
                                   target_transform=target_transformations, labels = y_train)

        print(f'Train dataset length: {self.train.__len__()}')

        self.validation = ASDataset(image_dir=IMAGE_PATH, subjects = test_subj, transform=transformation,
                                        target_transform=target_transformations, labels = y_test, train_mode=False)
                                        
        print(f'Validation dataset length: {self.validation.__len__()}')
        self.test = self.validation

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, drop_last = False)

    def val_dataloader(self):
        return DataLoader(self.validation, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, drop_last = False)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, drop_last = True)

