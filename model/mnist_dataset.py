# This file defines the MNIST dataset and the data loader. The data loader is 
# responsible for loading the MNIST dataset in a format that can be used by 
# the model.

import torch
from torch import Tensor

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_mnist_data_loader(batch_size, augmentations=None):
    if augmentations is None:
        augmentations = []
    transform_list = [transforms.ToTensor()] + augmentations + [
        transforms.Normalize((0.1307,), (0.3081,))
    ]
        
    # Define the transformation to be applied to the input data
    transform = transforms.Compose(transform_list)
    
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

def get_data_loader_from_image_folder(batch_size, path_to_folder):
    """
    Args:
    - batch_size: int, the batch size to be used
    - path_to_folder: str, the path to the folder containing the *.png images

    Returns:
    - data_loader: DataLoader, the data loader for the dataset
    """
    # Define the transformation to be applied to the input data
    transform = transforms.Compose([
        # grayscale image
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load the dataset from the folder
    data_loader = DataLoader(
        datasets.ImageFolder(path_to_folder, transform=transform),
        batch_size=batch_size, shuffle=False
    )
    
    return data_loader

class GaussianNoise(torch.nn.Module):
    """Add gaussian noise to images or videos.

    The input tensor is expected to be in [..., 1 or 3, H, W] format,
    where ... means it can have an arbitrary number of leading dimensions.
    Each image or frame in a batch will be transformed independently i.e. the
    noise added to each image will be different.

    The input tensor is also expected to be of float dtype in ``[0, 1]``.
    This transform does not support PIL images.

    Args:
        mean (float): Mean of the sampled normal distribution. Default is 0.
        sigma (float): Standard deviation of the sampled normal distribution. Default is 0.1.
        clip (bool, optional): Whether to clip the values in ``[0, 1]`` after adding noise. Default is True.
    """

    def __init__(self, mean: float = 0.0, sigma: float = 0.1, clip=True) -> None:
        super().__init__()
        self.mean = mean
        self.sigma = sigma
        self.clip = clip

    def forward(self, img: Tensor) -> Tensor:
        if not img.is_floating_point():
            raise ValueError(f"Input tensor is expected to be in float dtype, got dtype={image.dtype}")
        if self.sigma < 0:
            raise ValueError(f"sigma shouldn't be negative. Got {self.sigma}")

        noise = self.mean + torch.randn_like(img) * self.sigma
        noisy_img = img + noise

        if self.clip:
            noisy_img = torch.clamp(noisy_img, 0, 1)

        return noisy_img