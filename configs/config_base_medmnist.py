import torch
import dataset.data_processing_medmnist as data_processing_medmnist
import sl_evaluation_medmnist
import net_models_torch
import train_with_DA
import DA.data_augmentation_albumentations
import state_manager_torch
import mutations
import chromosomes
import evolution_mod_functions


DATA_FLAG = 'breastmnist'
RESNET_FLAG = 'resnet50' # Mude para 'resnet50' para o ResNet50
NUM_CLASSES_MEDMNIST = 2 # 2 classes para BreastMNIST

config = {}


config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device {config['device']}")
config['seeds'] = range(5)
config['seed'] = config['seeds'][0]
config['base_experiment_name'] = f"optimize_do_{DATA_FLAG}_{RESNET_FLAG}" 
config['experiment_name'] = config['base_experiment_name']
config['output_csv_folder'] = "output_csv" + "_" + config['base_experiment_name']

config['dataset'] = DATA_FLAG
config['dim'] = (28, 28, 1) # Dimensão 28x28
config['num_classes'] = NUM_CLASSES_MEDMNIST # 2 classes
config['num_classes_downstream'] = config['num_classes'] # 2 classes
config['cache_folder'] = f"cache_{DATA_FLAG}_torch"
config['delete_cache'] = False 


config['load_dataset_func'] = data_processing_medmnist.load_dataset
config['data_loader_func'] = data_processing_medmnist.create_data_loaders
config['dataset_transforms'] = data_processing_medmnist.dataset_transforms

if RESNET_FLAG == 'resnet50':
    config['model'] = net_models_torch.TrainResNet50Simple
else:
    config['model'] = net_models_torch.TrainResNet18Simple

# Configurações de Treino SL
config['base_epochs'] = 20
config['epochs'] = config['base_epochs']
config['extended_epochs'] = 100
config['batch_size'] = 128
config['num_workers'] = 4
config['shuffle_dataset'] = True

# Funções de Avaliação (Apontando para o pipeline SL)
config['individual_evaluation_func'] = train_with_DA.train_and_evaluate_individual
config['model_evaluate_func'] = sl_evaluation_medmnist.evaluate_sl


# O EA.py espera um genótipo de 2 níveis [Pretext, Downstream]
# Nós definimos o Pretext (índice 0) como FIXO e VAZIO.
# Nós definimos o Downstream (índice 1) como OTIMIZADO (None).
config['fix_pretext_da'] = [] 
config['fix_downstream_da'] = None
config['evolution_type'] = "simultaneous" # Irá mutar apenas o índice 1


config['min_da_prob'] = 0.1
config['max_da_prob'] = 0.9
config['da_funcs'] = DA.data_augmentation_albumentations.da_funcs_probs(config['min_da_prob'], config['max_da_prob'], config['dim'][:2])
config['n_pr'] = 4 
config['max_chromosomes'] = 5 

config['stop_gen'] = 220
config['population_size'] = 5
config['start_gen'] = 1
config['start_parent'] = None
config['start_population'] = None
config['best_n'] = 5
config['best_individuals'] = None

config['create_da_func'] = chromosomes.random_da_func(len(config['da_funcs']))
config['create_pr'] = chromosomes.random_pr
config['create_chromosome'] = chromosomes.create_chromosome_2_levels(config['create_da_func'], config['create_pr'], config['n_pr'])
config['da_func_mutation'] = chromosomes.random_da_func(len(config['da_funcs']))
config['pr_mutation'] = chromosomes.random_pr_gaussian(0.1)
config['mutation'] = mutations.mutate_remove_change_add_seq(0.66, 0.33, 0.66)

config['state_folder'] = "states"
config['state_file'] = None
config['load_state'] = state_manager_torch.load_state
config['save_state'] = state_manager_torch.save_state
config['every_gen_state_reset'] = None
config['extended_isolated_run'] = False
config['current_run_generations'] = 0
config['max_generations_per_run'] = None
config['recalculate_best'] = True
config['check_memory_leaks'] = False


config['evolution_mods'] = {
    201: evolution_mod_functions.extended_gens
}

config['confusion_matrix_config'] = {
     'print_confusion_matrix': True,
     'confusion_matrix_folder': "output_confusion_matrix",
     'confusion_matrix_pretext_file': "confusion_matrix_pretext.txt",
     'confusion_matrix_downstream_file': "confusion_matrix_downstream.txt",
     'num_classes_pretext': 4, 
     'num_classes_downstream': config['num_classes_downstream'] 
}

config['save_models_folder'] = None
config['save_pretext_model'] = None
config['save_downstream_model'] = None
# config['save_models_folder'] = "models"
# config['save_pretext_model'] = config['base_experiment_name'] + "_pretext"
# config['save_downstream_model'] = config['base_experiment_name'] + "_downstream"