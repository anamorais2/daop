import os
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import DA.data_augmentation_albumentations as data_augmentation_albumentations
import configs.config_base as config

class EchographyFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root=root)
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        img = np.array(sample)
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=2)
        elif img.shape[-1] == 1:
            img = img.repeat(3, axis=-1)
        if self.transform is not None:
            transformed = self.transform(image=img)
            img = transformed['image']
        return img, torch.tensor(target)
    
def dataset_transforms():
    # Typical mean and standard deviation for ResNet (ImageNet pre-trained)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
  
    return [], [A.Normalize(mean=mean, std=std), ToTensorV2()]

def load_dataset(individual, config):
    
    if not os.path.exists(config['cache_folder']):
        print(f"Folder {config['cache_folder']} does not exist, creating it")
        os.makedirs(config['cache_folder'])
        print(f"Created {config['cache_folder']}/")
        
    
    
    data_root = config.get('data_root_path', './data') 
    current_fold = config.get('fold_name', 'Fold1')     
    
    train_dir = os.path.join(data_root, current_fold, 'train')
    val_dir = os.path.join(data_root, current_fold, 'val')
    test_dir = os.path.join(data_root, current_fold, 'test')
    

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Testing directory not found: {test_dir}")
    
    if not os.path.exists(val_dir):
        print(f"Validation directory not found: {val_dir}. Proceeding without it.")

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

 
    transform_train = A.Compose(transforms_base + sl_augs + transforms_end)
    transform_eval = A.Compose(transforms_base + transforms_end)

    # 5. Create the Datasets
    trainset = EchographyFolder(root=train_dir, transform=transform_train)
    valset = EchographyFolder(root=val_dir, transform=transform_eval)
    testset = EchographyFolder(root=test_dir, transform=transform_eval)
    
    print(f"Classes found: {trainset.classes}")
    print(f"Images: Train={len(trainset)}, Val={len(valset)}, Test={len(testset)}")
    
    return trainset, valset, testset


def create_data_loaders(trainset, valset, testset, config):
    print("Creating data loaders (Thesis SL Mode)")
    
    batch_size = config.get('batch_size', 32)

    trainloader = DataLoader(
        trainset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=config['num_workers'], 
        pin_memory=True
    )
    
    valloader = DataLoader(
        valset,
        batch_size=batch_size,
        shuffle=False,
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

    return trainloader, valloader, testloader