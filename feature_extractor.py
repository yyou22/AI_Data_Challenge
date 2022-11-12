import os
import glob
import torch
import torch.nn as nn
import torchvision.models as models

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
        self.files = glob.glob(os.path.join(folder, '*.jpg'))
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.folder, "frame_" + str(idx) + ".jpg")
        img = Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return im
    def __len__(self):
        return len(self.files)