import torch
from torch.utils import data
from torchvision.datasets.utils import download_and_extract_archive,check_integrity
from torchvision.transforms import Compose, Normalize, ToTensor
import warnings
from PIL import Image
import os
from loader.mnist import MNIST
from loader.fmnist import FashionMNIST
import numpy as np

class OE_MNIST_train(data.Dataset):
    def __init__(self,root,num=100,transform = None):
        np.random.seed(0)
        mnist = MNIST(root)
        x_mnist = mnist.data
        y_mnist = mnist.targets
        fmnist = FashionMNIST(root)
        x_fmnist = fmnist.data
        temp = np.arange(x_fmnist.shape[0])
        temp = np.random.choice(temp,num)
        x_fmnist = x_fmnist[temp]
        y_fmnist = np.ones(x_fmnist.shape[0],dtype=int)*-1
        self.data = np.concatenate((x_mnist,x_fmnist))
        self.target = np.concatenate((y_mnist,y_fmnist))
        self.transform = transform

    def __getitem__(self, index):
        img = self.data[index]/255.0
        label = self.target[index]
        if self.transform != None:
            img = self.transform(img).type(torch.FloatTensor)
        return img, label

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    oe = OE_MNIST_train('./dataset/')
    a = oe.__getitem__(0)