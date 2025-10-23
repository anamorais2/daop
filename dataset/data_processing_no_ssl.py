import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import os

import rotnet_torch
import dataset.data_processing_no_ssl

import albumentations as A
from albumentations.pytorch import ToTensorV2
#from albumentations_dataset_wrappers.cifar10 import CIFAR10Albumentations
import DA.data_augmentation_albumentations as data_augmentation_albumentations

def dataset_transforms():
    return [], [A.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]), ToTensorV2()]

def dataset_transforms_no_norm():
    return [], [ToTensorV2()]

def load_dataset(individual, config):
    if not os.path.exists(config['cache_folder']):
        print(f"Folder {config['cache_folder']} does not exist, creating it")
        os.makedirs(config['cache_folder'])
        print(f"Created {config['cache_folder']}/")

    transforms_before_augs, transforms_after_augs = config['dataset_transforms']()

    individual = data_augmentation_albumentations.map_augments(individual[1], config)
    print(individual)

    transform = A.Compose(transforms_before_augs + transforms_after_augs)

    transform_augs = A.Compose(transforms_before_augs + individual + transforms_after_augs)

    if os.path.exists(os.path.join(config['cache_folder'], config['dataset_file_name'])):
        # if the dataset is already downloaded, load it directly from cache
        print(f"Loading dataset from {config['cache_folder']}/")
        trainset= CIFAR10Albumentations(root=config['cache_folder'], train=True, 
            download=False, transform=transform_augs)
        testset= CIFAR10Albumentations(root=config['cache_folder'], train=False, 
            download=False, transform=transform)
        print(f"Dataset loaded from {config['cache_folder']}/")
        return trainset, testset
    else:
        print(f"Downloading dataset to {config['cache_folder']}/")
        trainset = CIFAR10Albumentations(root=config['cache_folder'], train=True, 
            download=False, transform=transform_augs)
        testset = CIFAR10Albumentations(root=config['cache_folder'], train=False, 
            download=False, transform=transform)
        print(f"Dataset downloaded to {config['cache_folder']}/")
        return trainset, testset
    


def create_data_loaders(trainset, testset, config):
    print("Creating data loaders")

    trainloader= torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'], shuffle=True, 
                                                      num_workers=config['num_workers'], pin_memory=True)

    testloader= torch.utils.data.DataLoader(testset, batch_size=config['batch_size'], shuffle=False, 
                                                      num_workers=config['num_workers'], pin_memory=True)


    print("Data loaders created")

    return trainloader, testloader
    
