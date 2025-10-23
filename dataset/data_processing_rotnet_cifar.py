import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import os

import rotnet_torch
import DA.data_augmentation_albumentations as data_augmentation_albumentations

import albumentations as A
from albumentations.pytorch import ToTensorV2
#from albumentations_dataset_wrappers.cifar10 import CIFAR10Albumentations

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

    transform = A.Compose(transforms_before_augs + transforms_after_augs)

    pretext_augs = data_augmentation_albumentations.map_augments(individual[0], config)
    print(f"Pretext augs: {pretext_augs}")
    transform_pretext_augs = A.Compose(transforms_before_augs + pretext_augs + transforms_after_augs)

    downstream_augs = data_augmentation_albumentations.map_augments(individual[1], config)
    print(f"Downstream augs: {downstream_augs}")
    transform_downstream_augs = A.Compose(transforms_before_augs + downstream_augs + transforms_after_augs)

    if os.path.exists(os.path.join(config['cache_folder'], config['dataset_file_name'])):
        # if the dataset is already downloaded, load it directly from cache
        print(f"Loading dataset from {config['cache_folder']}/")
        trainset_pretext = CIFAR10Albumentations(root=config['cache_folder'], train=True, 
            download=False, transform=transform_pretext_augs)
        trainset_downstream = CIFAR10Albumentations(root=config['cache_folder'], train=True, 
            download=False, transform=transform_downstream_augs)
        testset_pretext = CIFAR10Albumentations(root=config['cache_folder'], train=False, 
            download=False, transform=transform)
        testset_downstream = CIFAR10Albumentations(root=config['cache_folder'], train=False, 
            download=False, transform=transform)
        print(f"Dataset loaded from {config['cache_folder']}/")
        return trainset_pretext, trainset_downstream, testset_pretext, testset_downstream
    else:
        print(f"Downloading dataset to {config['cache_folder']}/")
        trainset_pretext = CIFAR10Albumentations(root=config['cache_folder'], train=True, 
            download=True, transform=transform_pretext_augs)
        trainset_downstream = CIFAR10Albumentations(root=config['cache_folder'], train=True, 
            download=False, transform=transform_downstream_augs)
        testset_pretext = CIFAR10Albumentations(root=config['cache_folder'], train=False, 
            download=True, transform=transform)
        testset_downstream = CIFAR10Albumentations(root=config['cache_folder'], train=False, 
            download=False, transform=transform)
        print(f"Dataset downloaded to {config['cache_folder']}/")
        return trainset_pretext, trainset_downstream, testset_pretext, testset_downstream
    


def create_data_loaders(trainset_pretext, trainset_downstream, testset_pretext, testset_downstream, config):
    print("Creating data loaders")

    if config['rotations_on_cuda'] and config['device'] == torch.device('cuda'):
        trainloader_pretext = torch.utils.data.DataLoader(trainset_pretext, batch_size=config['pretext_batch_size'](), shuffle=config['shuffle_dataset'], 
                                                        collate_fn=rotnet_torch.rotnet_collate_fn_cuda, num_workers=config['num_workers'])

        trainloader_downstream = torch.utils.data.DataLoader(trainset_downstream, batch_size=config['downstream_batch_size'](), shuffle=config['shuffle_dataset'], 
                                                        num_workers=config['num_workers'])
        
        testloader_pretext = torch.utils.data.DataLoader(testset_pretext, batch_size=config['pretext_batch_size'](), shuffle=False, 
                                                        collate_fn=rotnet_torch.rotnet_collate_fn_cuda, num_workers=config['num_workers'])

        testloader_downstream = torch.utils.data.DataLoader(testset_downstream, batch_size=config['downstream_batch_size'](), shuffle=False, 
                                                        num_workers=config['num_workers'])
        
        print("Data loaders created (rotations on cuda mode)")
    
    else:
        trainloader_pretext = torch.utils.data.DataLoader(trainset_pretext, batch_size=config['pretext_batch_size'](), shuffle=config['shuffle_dataset'], 
                                                      collate_fn=rotnet_torch.rotnet_collate_fn, num_workers=config['num_workers'], pin_memory=True)

        trainloader_downstream = torch.utils.data.DataLoader(trainset_downstream, batch_size=config['downstream_batch_size'](), shuffle=config['shuffle_dataset'], 
                                                        num_workers=config['num_workers'], pin_memory=True)
        
        testloader_pretext = torch.utils.data.DataLoader(testset_pretext, batch_size=config['pretext_batch_size'](), shuffle=False, 
                                                        collate_fn=rotnet_torch.rotnet_collate_fn, num_workers=config['num_workers'], pin_memory=True)

        testloader_downstream = torch.utils.data.DataLoader(testset_downstream, batch_size=config['downstream_batch_size'](), shuffle=False, 
                                                        num_workers=config['num_workers'], pin_memory=True)


        print("Data loaders created (default mode)")

    return trainloader_pretext, trainloader_downstream, testloader_pretext, testloader_downstream
    
