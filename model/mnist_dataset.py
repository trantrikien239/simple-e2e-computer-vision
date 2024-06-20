# This file defines the MNIST dataset and the data loader. The data loader is 
# responsible for loading the MNIST dataset in a format that can be used by 
# the model.

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_mnist_data_loader(batch_size):
    # Define the transformation to be applied to the input data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load the training and test datasets
    train_loader = DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        datasets.MNIST('data', train=False, transform=transform),
        batch_size=batch_size, shuffle=True
    )
    
    return train_loader, test_loader