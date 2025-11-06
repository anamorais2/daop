import os
import torch
import torchvision.transforms as transforms
import albumentations as A
import numpy as np # Importar NumPy para manipulação de array
from albumentations.pytorch import ToTensorV2
from medmnist import INFO, Evaluator, dataset as medmnist_dataset

import rotnet_torch
import DA.data_augmentation_albumentations as data_augmentation_albumentations

# ==============================================================================
# 1. Definições e Wrapper do Dataset
# ==============================================================================

DATA_FLAG = 'breastmnist'
INFO_BREAST = INFO[DATA_FLAG]
DataClass = getattr(medmnist_dataset, INFO_BREAST['python_class'])


class BreastMNISTAlbumentations(DataClass):
    
    def __init__(self, transform=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transform = transform
        # Força o carregamento no formato HxWxC para compatibilidade com Albumentations
        self.pil = True 

    def __getitem__(self, index):
        # MedMNIST retorna: (imagem, rótulo).
        img, target = self.imgs[index], self.labels[index]

        # CORREÇÃO CRÍTICA: FORÇAR 3 CANAIS (H, W, 3) PARA ALBUMENTATIONS
        # O Albumentations requer (H, W, C) para transformações como Equalize, Rotate, etc.
        
        # 1. Verificar se a imagem tem apenas 2 dimensões (H, W) ou 3 dimensões (H, W, 1)
        if len(img.shape) == 2:
            # Se for (H, W), empilhamos 3 vezes ao longo de uma nova dimensão (H, W, 3)
            img = np.stack([img] * 3, axis=2)
        elif img.shape[-1] == 1:
            # Se for (H, W, 1), replicamos o canal 3 vezes (H, W, 3)
            img = img.repeat(3, axis=-1)
        
        # Agora img é garantidamente (H, W, 3)

        if self.transform is not None:
            # Aplica as transformações Albumentations, que agora recebem 3 canais
            transformed = self.transform(image=img)
            img = transformed['image']

        # O rótulo é um tensor (1 classe), converte para escalar para RotNet
        return img, torch.tensor(target).squeeze()


def dataset_transforms():
    """
    Define as transformações de normalização. 
    Usaremos a normalização padrão do MedMNIST (0.5, 0.5) para um canal, 
    replicada 3 vezes para o RotNet/ResNet (que esperam 3 canais).
    """
    # BreastMNIST: valores médios e desvios padrão (para 3 canais replicados)
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    return [], [A.Normalize(mean=mean, std=std), ToTensorV2()]


def load_dataset(individual, config):
    if not os.path.exists(config['cache_folder']):
        print(f"Folder {config['cache_folder']} does not exist, creating it")
        os.makedirs(config['cache_folder'])
        print(f"Created {config['cache_folder']}/")

    transforms_before_augs, transforms_after_augs = config['dataset_transforms']()
    
    # 1. Transformação Base (para teste/validação - sem aumentos)
    transform = A.Compose(transforms_before_augs + transforms_after_augs)

    # 2. Transformação Pretext (com aumentos fixos do RotNet)
    pretext_augs = data_augmentation_albumentations.map_augments(individual[0], config)
    print(f"Pretext augs: {pretext_augs}")
    transform_pretext_augs = A.Compose(transforms_before_augs + pretext_augs + transforms_after_augs)

    # 3. Transformação Downstream (com aumentos otimizados pelo DAOP)
    downstream_augs = data_augmentation_albumentations.map_augments(individual[1], config)
    print(f"Downstream augs: {downstream_augs}")
    transform_downstream_augs = A.Compose(transforms_before_augs + downstream_augs + transforms_after_augs)

    # Carregamento dos Datasets (usando o novo Wrapper)
    print(f"Loading dataset {DATA_FLAG} from {config['cache_folder']}/")

    # A lógica de cache/download do MedMNIST é mais simples que o CIFAR
    download_needed = not os.path.exists(os.path.join(config['cache_folder'], DATA_FLAG + '.npz'))
    
    # TREINO (Pretext e Downstream)
    trainset_pretext = BreastMNISTAlbumentations(root=config['cache_folder'], split='train', 
        download=download_needed, transform=transform_pretext_augs)
    trainset_downstream = BreastMNISTAlbumentations(root=config['cache_folder'], split='train', 
        download=False, transform=transform_downstream_augs)
    
    # TESTE (Pretext e Downstream)
    # Nota: MedMNIST tem split 'val' e 'test'. Usaremos 'test' para a avaliação final.
    testset_pretext = BreastMNISTAlbumentations(root=config['cache_folder'], split='test', 
        download=False, transform=transform)
    testset_downstream = BreastMNISTAlbumentations(root=config['cache_folder'], split='test', 
        download=False, transform=transform)
    
    print(f"Dataset {DATA_FLAG} loaded successfully.")
    return trainset_pretext, trainset_downstream, testset_pretext, testset_downstream


def create_data_loaders(trainset_pretext, trainset_downstream, testset_pretext, testset_downstream, config):
    
    print("Creating data loaders")

    # O DAOP já define a variável device
    is_cuda_active = config['rotations_on_cuda'] and config['device'] == torch.device('cuda')

    # Define a função collate_fn com base na disponibilidade do CUDA
    collate_fn = rotnet_torch.rotnet_collate_fn_cuda if is_cuda_active else rotnet_torch.rotnet_collate_fn
    
    # DataLoaders para Pretext (RotNet)
    trainloader_pretext = torch.utils.data.DataLoader(
        trainset_pretext, 
        batch_size=config['pretext_batch_size'](), 
        shuffle=config['shuffle_dataset'], 
        collate_fn=collate_fn, 
        num_workers=config['num_workers'], 
        pin_memory=True
    )
    testloader_pretext = torch.utils.data.DataLoader(
        testset_pretext, 
        batch_size=config['pretext_batch_size'](), 
        shuffle=False, 
        collate_fn=collate_fn, 
        num_workers=config['num_workers'], 
        pin_memory=True
    )

    # DataLoaders para Downstream (Classificação)
    trainloader_downstream = torch.utils.data.DataLoader(
        trainset_downstream, 
        batch_size=config['downstream_batch_size'](), 
        shuffle=config['shuffle_dataset'], 
        num_workers=config['num_workers'], 
        pin_memory=True
    )
    testloader_downstream = torch.utils.data.DataLoader(
        testset_downstream, 
        batch_size=config['downstream_batch_size'](), 
        shuffle=False, 
        num_workers=config['num_workers'], 
        pin_memory=True
    )

    mode = "cuda" if is_cuda_active else "default"
    print(f"Data loaders created ({mode} mode)")

    return trainloader_pretext, trainloader_downstream, testloader_pretext, testloader_downstream
    
    


# ==============================================================================
# 3. Funções de Carregamento de Dados (Simplificado para SL)
# ==============================================================================

def load_dataset_simple(individual, config):
    """
    Carrega apenas os datasets de Treino (com DA) e Teste (sem DA).
    """
    if not os.path.exists(config['cache_folder']):
        print(f"Folder {config['cache_folder']} does not exist, creating it")
        os.makedirs(config['cache_folder'])
        print(f"Created {config['cache_folder']}/")

    transforms_before_augs, transforms_after_augs = config['dataset_transforms']()
    
    # 1. Transformação Base (para teste/validação - sem aumentos)
    transform_test = A.Compose(transforms_before_augs + transforms_after_augs)

    # 2. Transformação de Treino (SL)
    
    augs_genotype = individual[0] # O genótipo (a lista de aumentos)
    sl_augs_list = [] # Inicializar a lista de aumentos

    # --- INÍCIO DA CORREÇÃO ---
    # Verificamos a estrutura do genótipo para garantir que map_augments recebe uma lista de listas
    
    if (len(augs_genotype) > 0 and 
        isinstance(augs_genotype[0], list) and 
        isinstance(augs_genotype[0][0], int)):
        # CASO 1: Genótipo SL (Supervised Learning) [ [Aug 1], [Aug 2], ... ]
        # Ex: [ [1, [params]], [37, [params]] ]
        sl_augs_list = augs_genotype
        print(f"SL Augs (from SL genotype): {sl_augs_list}")

    elif (len(augs_genotype) == 2 and 
          isinstance(augs_genotype[0], list) and 
          isinstance(augs_genotype[1], list)):
        # CASO 2: Genótipo DO (Downstream Opt) [ [Pretext_List], [Downstream_List] ]
        # Ex: [ [[0, [p]]], [[1, [p]], [37, [p]]] ]
        sl_augs_list = augs_genotype[1] # Usamos apenas a lista Downstream
        print(f"SL Augs (from DO genotype): {sl_augs_list}")
        
    elif (len(augs_genotype) == 2 and 
          isinstance(augs_genotype[0], int) and 
          isinstance(augs_genotype[1], list)):
        # CASO 3: Genótipo é uma ÚNICA AUG 
        # Ex: [ 1, [0.5,...] ]
        # Devemos envolvê-lo numa lista para map_augments:
        sl_augs_list = [ augs_genotype ]
        print(f"SL Augs (Single Aug wrapped): {sl_augs_list}")
    
    else:
        # Caso de fallback (pode ser uma lista vazia ou estrutura desconhecida)
        sl_augs_list = augs_genotype
        print(f"SL Augs (Fallback): {sl_augs_list}")
    
    # --- FIM DA CORREÇÃO ---

    sl_augs = data_augmentation_albumentations.map_augments(sl_augs_list, config)
    transform_train = A.Compose(transforms_before_augs + sl_augs + transforms_after_augs)

    # ... (O resto da função de carregamento: download, BreastMNISTAlbumentations, etc.)

    print(f"Loading dataset {DATA_FLAG} from {config['cache_folder']}/ (SL Mode)")
    download_needed = not os.path.exists(os.path.join(config['cache_folder'], DATA_FLAG + '.npz'))
    
    trainset = BreastMNISTAlbumentations(root=config['cache_folder'], split='train', 
        download=download_needed, transform=transform_train)
    
    testset = BreastMNISTAlbumentations(root=config['cache_folder'], split='test', 
        download=False, transform=transform_test)
    
    print(f"Dataset {DATA_FLAG} loaded successfully.")
    
    return trainset, testset


def create_data_loaders_simple(trainset, testset, config):
   
    # Usa o config['batch_size'] padrão (não pretext/downstream)
    batch_size = config.get('batch_size', 128) # Usar .get para segurança

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


    return trainloader, testloader
