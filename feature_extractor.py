import os
import glob
import torch
import torch.nn as nn
import torchvision.models as models
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        self.net = model
        # If you treat the model as a fixed feature extractor, disable the gradients and save some memory
        for p in self.net.parameters():
            p.requires_grad = False
        # Define which layers you are going to extract
        self.features = nn.Sequential(*list(self.net.children())[:-1])

    def forward(self, x):
        return self.features(x)

class ImageDataset(object):
    def __init__(self, folder, transforms=None):
        self.folder = folder
        self.transforms = transforms
        self.files = glob.glob(os.path.join(folder, '*.jpg'))
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.folder, "frame_" + str(idx + 1) + ".jpg")
        img = Image.open(img_path)

        imageId = idx + 1

        if self.transforms is not None:
            img = self.transforms(img)

        return img, imageId

    def __len__(self):
        return len(self.files)

class ImageDataset_tar(object):
    def __init__(self, folder, transforms=None):
        self.folder = folder
        self.transforms = transforms
        self.files = glob.glob(os.path.join(folder, '*.jpg'))

        self.csv_file_name = 'ROI1.csv'

        csv_file_path = os.path.join(
            self.folder, self.csv_file_name)

        self.csv_data = pd.read_csv(csv_file_path, sep=';', usecols=["Filename", "ClassId"])
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.folder, "frame_" + str(idx + 1) + ".jpg")
        img = Image.open(img_path)

        imageId = idx + 1

        if self.transforms is not None:
            img = self.transforms(img)

        classId = self.csv_data.iloc[idx, 1]

        return img, target, imageId

    def __len__(self):
        return len(self.files)