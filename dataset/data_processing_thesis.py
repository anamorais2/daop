import os
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import rotnet_torch
import DA.data_augmentation_albumentations as data_augmentation_albumentations
import configs.config_base as config

class ThesisAlbumentations(ImageFolder):

    def __init__(self, root, transform=None):
        super().__init__(root=root)
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path) 
        sample = np.array(sample)  
        
        if self.transform is not None:
            augmented = self.transform(image=sample)
            sample = augmented['image']
            
        return sample, target
    
def dataset_transforms():
    # Typical mean and standard deviation for ResNet (ImageNet pre-trained)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
  
    return [], [A.Normalize(mean=mean, std=std), ToTensorV2()]

def load_dataset(individual, config):
    
    
    data_root = config.get('data_root_path', './data') 
    current_fold = config.get('fold_name', 'Fold1')     
    
    train_dir = os.path.join(data_root, current_fold, 'Train')
    test_dir = os.path.join(data_root, current_fold, 'Test')
    # val_dir = os.path.join(data_root, current_fold, 'Val') # If you want to use validation

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    # Resize
    img_height = config['dim'][0]
    img_width = config['dim'][1]
    resize_transform = [A.Resize(height=img_height, width=img_width)]

    transforms_base, transforms_end = dataset_transforms()
    
    augs_genotype = individual[0]
    sl_augs_list = []
    
    if (len(augs_genotype) > 0 and isinstance(augs_genotype[0], list)):
        sl_augs_list = augs_genotype
    elif (len(augs_genotype) == 2 and isinstance(augs_genotype[0], int)):
         sl_augs_list = [augs_genotype]
    else:
        sl_augs_list = augs_genotype

    sl_augs = data_augmentation_albumentations.map_augments(sl_augs_list, config)
    
    print(f"--- Loading {current_fold} ---")
    print(f"Applied augmentations: {sl_augs}")

    # 4. Compose final transforms
    # Train: Resize -> Optimized Augmentations -> Normalize -> Tensor
    transform_train = A.Compose(resize_transform + transforms_base + sl_augs + transforms_end)
    
    # Test: Resize -> Normalize -> Tensor (NO Augmentations)
    transform_test = A.Compose(resize_transform + transforms_base + transforms_end)

    # 5. Create the Datasets
    trainset = ThesisAlbumentations(root=train_dir, transform=transform_train)
    testset = ThesisAlbumentations(root=test_dir, transform=transform_test)
    
    print(f"Classes found: {trainset.classes}")
    return trainset, testset


def create_data_loaders(trainset, testset, config):
    print("Creating data loaders (Thesis SL Mode)")
    
    batch_size = config.get('batch_size', 128)

    trainloader = DataLoader(
        trainset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=config['num_workers'], 
        pin_memory=True
    )
    
    testloader = DataLoader(
        testset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=config['num_workers'], 
        pin_memory=True
    )

    return trainloader, testloader