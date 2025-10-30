import configs.config_base_resnet18 as config_base

import dataset.data_processing_breastmnist as data_processing_medmnist
import DA.data_augmentation_albumentations as data_augmentation_albumentations
import net_models_torch
import rotnet_torch

# ==============================================================================
# 1. Parâmetros Específicos do BreastMNIST
# ==============================================================================
DATA_FLAG = 'breastmnist'
RESNET_FLAG = 'resnet18'
NUM_CLASSES_MEDMNIST = 2 
ROTNET_DA = [[0, [1.0, 0.2, 0.2, 0.2, 0.2]], [1, [0.5, 0.5, 0.5, 0.5, 0.5]]]

config = config_base.config

if config['model'] == net_models_torch.TrainResNet18:
    RESNET_FLAG = 'resnet18'
else:
    RESNET_FLAG = 'resnet50'

# ==============================================================================
# 2. Configurações Gerais do Experimento
# ==============================================================================
config['base_experiment_name'] = f"optimize_do_{DATA_FLAG}_{RESNET_FLAG}"
config['experiment_name'] = config['base_experiment_name']
config['output_csv_folder'] = "output_csv" + "_" + config['base_experiment_name']

# Fixar DA para o RotNet (Pretext) e otimizar para Downstream (DO)
config['fix_pretext_da'] = ROTNET_DA
config['fix_downstream_da'] = None 

# ==============================================================================
# 3. Configurações de Dados: Substituição do CIFAR-10 por BreastMNIST
# ==============================================================================
config['dataset'] = DATA_FLAG
config['dim'] = (28, 28, 1) 
config['num_classes'] = NUM_CLASSES_MEDMNIST
config['num_classes_downstream'] = config['num_classes']
config['cache_folder'] = f"cache_{DATA_FLAG}_torch"

# ATENÇÃO: SUBSTITUIR ESTAS FUNÇÕES
# Estas funções devem estar no seu novo ficheiro data_processing_medmnist.py
config['load_dataset_func'] = data_processing_medmnist.load_dataset
config['data_loader_func'] = data_processing_medmnist.create_data_loaders
config['dataset_transforms'] = data_processing_medmnist.dataset_transforms


config['da_funcs'] = data_augmentation_albumentations.da_funcs_probs(
    config['min_da_prob'], config['max_da_prob'], config['dim'][:2]
)

# ==============================================================================
# 4. Configurações do Modelo e Treino
# ==============================================================================
config['model'] = net_models_torch.TrainResNet18 # Manter, mas ResNet pode ser pesado para 28x28
config['batch_size'] = 128
config['pretext_batch_size'] = lambda: config['batch_size'] * 2
config['downstream_batch_size'] = lambda: config['batch_size']
config['num_workers'] = 4
