from logging import config
import os
import torch
import albumentations as A
import numpy as np 
from albumentations.pytorch import ToTensorV2
from medmnist import INFO, Evaluator, dataset as medmnist_dataset
import configs.config_base_val as config
import DA.data_augmentation_albumentations as data_augmentation_albumentations

DATA_FLAG = config.DATA_FLAG
INFO = INFO[DATA_FLAG]
DataClass = getattr(medmnist_dataset, INFO['python_class'])


class MEDMNISTAlbumentations(DataClass):
    
    def __init__(self, transform=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transform = transform
        self.pil = True 

    def __getitem__(self, index):
        # MedMNIST returns: (image, label).
        img, target = self.imgs[index], self.labels[index]

        # Albumentations requires (H, W, C) for transforms like Equalize, Rotate, etc.
        
        # 1. Check if image has only 2 dimensions (H, W) or 3 dims (H, W, 1)
        if len(img.shape) == 2:
            # If (H, W), stack it 3 times along a new dimension to get (H, W, 3)
            img = np.stack([img] * 3, axis=2)
        elif img.shape[-1] == 1:
            # If (H, W, 1), replicate the channel 3 times (H, W, 3)
            img = img.repeat(3, axis=-1)
        

        if self.transform is not None:
            transformed = self.transform(image=img)
            img = transformed['image']

        return img, torch.tensor(target).squeeze()
    


def dataset_transforms():
   
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    return [], [A.Normalize(mean=mean, std=std), ToTensorV2()]

def load_dataset_simple(individual, config):
    
    if not os.path.exists(config['cache_folder']):
        print(f"Folder {config['cache_folder']} does not exist, creating it")
        os.makedirs(config['cache_folder'])
        print(f"Created {config['cache_folder']}/")

    transforms_before_augs, transforms_after_augs = config['dataset_transforms']()
    
    transform_eval = A.Compose(transforms_before_augs + transforms_after_augs)

  
    if config.get('fix_pretext_da') is not None:
        # DO mode: use ONLY downstream (individual[1])
        augs_genotype = individual[1]
        print(f"DO Mode detected - using downstream augs from individual[1]")
    else:
        # Pure SL mode: use individual[0]
        augs_genotype = individual[0]
        print(f"SL Mode detected - using genotype from individual[0]")
    
   
    sl_augs_list = []
    
    if not augs_genotype or len(augs_genotype) == 0:
        # Empty list - no augmentations
        sl_augs_list = []
        print(f"SL Augs: Empty (no augmentations)")
        
    elif isinstance(augs_genotype[0], list):
        # Augmentations list [[id, params], [id, params], ...]
        if len(augs_genotype[0]) == 2 and isinstance(augs_genotype[0][0], int):
            sl_augs_list = augs_genotype
            print(f"SL Augs (list of augs): {sl_augs_list}")
        else:
            # Unexpected case - fallback
            sl_augs_list = augs_genotype
            print(f"SL Augs (nested list - fallback): {sl_augs_list}")
            
    elif len(augs_genotype) == 2 and isinstance(augs_genotype[0], int):
        # Single aug [id, params] - wrap into list
        sl_augs_list = [augs_genotype]
        print(f"SL Augs (single aug wrapped): {sl_augs_list}")
        
    else:
        # Generic fallback
        sl_augs_list = augs_genotype
        print(f"SL Augs (fallback): {sl_augs_list}")

    # Map augmentations using the mapping function
    sl_augs = data_augmentation_albumentations.map_augments(sl_augs_list, config)
    transform_train = A.Compose(transforms_before_augs + sl_augs + transforms_after_augs)

  
    print(f"Loading dataset {DATA_FLAG} from {config['cache_folder']}/ (SL Mode)")
    download_needed = not os.path.exists(os.path.join(config['cache_folder'], DATA_FLAG + '.npz'))
    
    trainset = MEDMNISTAlbumentations(root=config['cache_folder'], split='train', 
                                      download=download_needed, transform=transform_train)
    
    valset = MEDMNISTAlbumentations(root=config['cache_folder'], split='val', 
                                    download=False, transform=transform_eval)
    
    testset = MEDMNISTAlbumentations(root=config['cache_folder'], split='test', 
                                     download=False, transform=transform_eval)
    
    print(f"Dataset {DATA_FLAG} loaded successfully.")
    
    return trainset, valset, testset

def create_data_loaders_simple(trainset, valset, testset, config):

    print("Creating DataLoaders (Train, Val, Test)...")
    batch_size = config.get('batch_size', 128)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, 
        num_workers=config['num_workers'], pin_memory=True
    )
    
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=False, 
        num_workers=config['num_workers'], pin_memory=True
    )
    
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, 
        num_workers=config['num_workers'], pin_memory=True
    )

    return trainloader, valloader, testloader
