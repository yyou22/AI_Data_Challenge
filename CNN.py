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
import numpy as np

from feature_extractor import FeatureExtractor
from feature_extractor import ImageDataset

parser = argparse.ArgumentParser(description='PyTorch ResNet feature extracting')
parser.add_argument('--no-cuda', action='store_true', default=False,
					help='disables CUDA training')
parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
					help='input batch size for testing (default: 200)')

args = parser.parse_args()

# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

testset = ImageDataset(
	folder='/content/croped_framed/1/',
	transforms=None
)

test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

def rep(model, device, test_loader):
	model.eval()

	#feature list
	features = []

	for data in test_loader:
		data = data.to(device)
		X = Variable(data)

		feat = model(X).reshape(X.shape[0], 2048)
		features.extend(feat.cpu().detach().numpy())

	features = np.array(features)

	return features;

def main():
	model = resnet101(weights=ResNet101_Weights.DEFAULT)
	model = model.to(device)

	#get backbone of CNN
	backbone = FeatureExtractor(model)
	backbone = backbone.to(device)

	features = rep(backbone, device, testset)

if __name__ == '__main__':
	main()
