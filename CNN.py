from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet101, ResNet101_Weights
from feature_extractor import FeatureExtractor
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch ResNet feature extracting')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

args = parser.parse_args()

# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

def main():
    model = resnet101(weights=ResNet101_Weights.DEFAULT)
    model = model.to(device)

    #get backbone of CNN
    backbone = FeatureExtractor(model)
    backbone = backbone.to(device)


if __name__ == '__main__':
    main()