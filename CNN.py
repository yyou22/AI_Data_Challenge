from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.models import resnet101, ResNet101_Weights
import numpy as np
import pandas as pd

from feature_extractor import FeatureExtractor
from feature_extractor import ImageDataset

from sklearn.cluster import KMeans
from IPython.display import Image

from collections import Counter

parser = argparse.ArgumentParser(description='PyTorch ResNet feature extracting')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
                    help='input batch size for testing (default: 200)')
parser.add_argument('--model-path',
                    default='./model-check/model_roi5.pt', #here
                    help='model for white-box attack evaluation')

args = parser.parse_args()

# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# set up data loader
transform_test = transforms.Compose([
    transforms.Resize((244, 244)),
    transforms.ToTensor(),
])

testset = ImageDataset(
    folder='/content/croped_framed/croped_framed/5/', #here
    #folder='/content/croped_framed/2/2/',
    transforms=transform_test
)

test_loader = torch.utils.data.DataLoader(testset, batch_size=200, shuffle=False, **kwargs)

def k_means_(img_features):
    k = 2
    clusters = KMeans(k, random_state = 40)
    clusters.fit(img_features)
    knn_labels = clusters.labels_

    keys_ = Counter(knn_labels).keys()
    val_ = Counter(knn_labels).values()

    print(knn_labels)
    print(keys_)
    print(val_)

    return knn_labels

def rep(model, device, test_loader):
    model.eval()

    #feature list
    features = []
    imgIDs = []

    for data, imgID in test_loader:
        data = data.to(device)
        imgID = imgID.to(device)

        X = Variable(data)
        y = Variable(imgID)

        feat = model(X).reshape(X.shape[0], 2048)
        features.extend(feat.cpu().detach().numpy())
        imgIDs.extend(y.data.cpu().detach().numpy())

    features = np.array(features)
    imgIDs = np.array(imgIDs)

    return features, imgIDs;

def main():
    model = resnet101(weights=ResNet101_Weights.DEFAULT)
    #model = resnet101()
    #model.fc = nn.Linear(2048, 2)
    #model = model.to(device)
    #model.load_state_dict(torch.load(args.model_path))

    #get backbone of CNN
    backbone = FeatureExtractor(model)
    backbone = backbone.to(device)

    features,imgIDs = rep(backbone, device, test_loader)

    knn_labels = k_means_(features)

    image_cluster = pd.DataFrame(imgIDs,columns=['image'])

    image_cluster["knn"] = knn_labels

    image_cluster.to_csv("output5.csv", index=False) #here

if __name__ == '__main__':
    main()
