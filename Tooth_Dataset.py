import torch.utils.data as data

from os import listdir
from os.path import join
from PIL import Image
import cv2
import os
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    return y


class ToothDataset(Dataset):
    """Segmented Tooth dataset."""

    def __init__(self, dataframe, root_dir, transform=None):
        """
        Args:
            dataframe (pd.dataframe): Person ID's dataframes.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.tooths_frame = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.tooths_frame)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.tooths_frame.iloc[idx, 1])
        
        image = cv2.imread(img_name)
        

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        otherinfs = self.tooths_frame.iloc[idx, 4]
        otherinf = ""
        if otherinfs:
            otherinf = 1
        else:
            otherinf = 0
        otherinfs = np.array([otherinf])
        otherinfs = otherinfs.astype('long').reshape(-1, 1)
        personid = self.tooths_frame.iloc[idx, 2]
        personid = np.array([personid])
        personid = personid.astype('long').reshape(-1, 1)
        toothid = self.tooths_frame.iloc[idx, 3]
        toothid = np.array([toothid])
        toothid = toothid.astype('long').reshape(-1, 1)
        
        if self.transform:
            image = self.transform(image)
           

        sample = {'image': image, 'other': otherinf, 'personid': personid,'toothid':toothid}
        return sample
