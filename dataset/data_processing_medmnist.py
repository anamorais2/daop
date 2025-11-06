import os
import torch
import torchvision.transforms as transforms
import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from medmnist import INFO, Evaluator, dataset as medmnist_dataset
import DA.data_augmentation_albumentations as data_augmentation_albumentations



class MedMNISTWrapperBase:
   
    def __init__(self, *args, **kwargs):
        
        self.albumentations_transform = kwargs.pop('transform', None) 
        kwargs['transform'] = None 
        
        kwargs.pop('data_flag', None) # Remove 'data_flag' de kwargs
            
        super().__init__(*args, **kwargs)
        self.pil = True 

    def __getitem__(self, index):
        img, target = self.imgs[index], self.labels[index]

        # (Lógica de replicação de canal 1->3)
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=2) 
        elif img.shape[-1] == 1:
            img = img.repeat(3, axis=-1)
        
        if self.albumentations_transform is not None:
            transformed = self.albumentations_transform(image=img) 
            img = transformed['image']

        return img, torch.tensor(target).squeeze()

    
def dataset_transforms():
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    return [], [A.Normalize(mean=mean, std=std), ToTensorV2()]

    
def load_dataset(individual, config):
    data_flag = config.get('dataset', 'breastmnist') 
    
    if not os.path.exists(config['cache_folder']):
        os.makedirs(config['cache_folder'])

    transforms_before_augs, transforms_after_augs = config['dataset_transforms']()
    transform_test = A.Compose(transforms_before_augs + transforms_after_augs)

    # (Lógica de verificação do genótipo - sl_augs_list)
    augs_genotype = individual[0]
    sl_augs_list = []
    if (len(augs_genotype) > 0 and isinstance(augs_genotype[0], list) and isinstance(augs_genotype[0][0], int)):
        sl_augs_list = augs_genotype
        print(f"SL Augs (from SL genotype): {sl_augs_list}")
    elif (len(augs_genotype) == 2 and isinstance(augs_genotype[0], list) and isinstance(augs_genotype[1], list)):
        sl_augs_list = augs_genotype[1]
        print(f"SL Augs (from DO genotype): {sl_augs_list}")
    elif (len(augs_genotype) == 2 and isinstance(augs_genotype[0], int) and isinstance(augs_genotype[1], list)):
        sl_augs_list = [ augs_genotype ]
        print(f"SL Augs (Single Aug wrapped): {sl_augs_list}")
    else:
        sl_augs_list = augs_genotype
        print(f"SL Augs (Fallback): {sl_augs_list}")

    sl_augs = data_augmentation_albumentations.map_augments(sl_augs_list, config)
    transform_train = A.Compose(transforms_before_augs + sl_augs + transforms_after_augs)

    info = INFO[data_flag]
    DataClass = getattr(medmnist_dataset, info['python_class'])
    
    class SpecificMedMNISTWrapper(MedMNISTWrapperBase, DataClass):
         def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    print(f"Loading dataset {data_flag} from {config['cache_folder']}/ (SL Mode)")
    download_needed = not os.path.exists(os.path.join(config['cache_folder'], data_flag + '.npz'))
    
    trainset = SpecificMedMNISTWrapper(data_flag=data_flag, split='train', 
        download=download_needed, transform=transform_train, root=config['cache_folder'])
    
    testset = SpecificMedMNISTWrapper(data_flag=data_flag, split='test', 
        download=False, transform=transform_test, root=config['cache_folder'])
    
    print(f"Dataset {data_flag} loaded successfully.")
    
    return trainset, testset


def create_data_loaders(trainset, testset, config):
    print("Creating data loaders (SL Mode)")
    batch_size = config.get('batch_size', 128)

    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=batch_size, 
        shuffle=config['shuffle_dataset'], 
        num_workers=config['num_workers'], 
        pin_memory=True
    )
    testloader = torch.utils.data.DataLoader(
        testset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=config['num_workers'], 
        pin_memory=True
    )
    print("Data loaders created (SL mode)")
    
    return trainloader, testloader