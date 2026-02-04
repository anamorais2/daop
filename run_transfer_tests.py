import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import pandas as pd
import ast
import os
import glob
import sys
import time
import gc 

import configs.config_base_val_transfer as config_base
import net_models_torch
import sl_evaluation_medmnist_val as sl_eval
import dataset.data_processing_medmnist_val as data_loader
import DA.data_augmentation_albumentations as da_lib 




DATASETS = ['breastmnist', 'dermamnist', 'pneumoniamnist', 'organcmnist']
MODELS = ['resnet18', 'resnet50']

EXPERIMENTS_LIST = []

#Test 2: Same Dataset, Different Model
for dataset in DATASETS:
    
    EXPERIMENTS_LIST.append({
        "test_name": f"TEST2_{dataset}_R18_to_R50",
        "source_folder": f"VAL_output_csv_VAL_optimize_do_{dataset}_resnet18",
        "source_dataset": dataset,
        "target_dataset": dataset,
        "target_model": "resnet50",
        "resolution": 28,
        "seeds": [1, 2, 3]
    })
    

    if dataset == 'pneumoniamnist':
        print(f"Skipping TEST2 for {dataset} ResNet50 (Source missing)")
    else:
        EXPERIMENTS_LIST.append({
            "test_name": f"TEST2_{dataset}_R50_to_R18",
            "source_folder": f"VAL_output_csv_VAL_optimize_do_{dataset}_resnet50",
            "source_dataset": dataset,
            "target_dataset": dataset,
            "target_model": "resnet18",
            "resolution": 28,
            "seeds": [1, 2, 3]
        })

# Test 3: Different Dataset, Different Model
for src_dataset in DATASETS:
    for src_model in MODELS:
        
        if src_dataset == 'pneumoniamnist' and src_model == 'resnet50':
            print(f"Skipping TEST3 experiments with source: {src_dataset} + {src_model}")
            continue

        src_folder = f"VAL_output_csv_VAL_optimize_do_{src_dataset}_{src_model}"
        
        for tgt_dataset in DATASETS:
            if src_dataset == tgt_dataset:
                continue
            
            for tgt_model in MODELS:
                test_id = f"TEST3_Src_{src_dataset[:6]}{src_model[-2:]}_Tgt_{tgt_dataset[:6]}{tgt_model[-2:]}"
                
                EXPERIMENTS_LIST.append({
                    "test_name": test_id,
                    "source_folder": src_folder,
                    "source_dataset": src_dataset,
                    "target_dataset": tgt_dataset,
                    "target_model": tgt_model,
                    "resolution": 28,
                    "seeds": [1] 
                })



'''
SOURCE_DATASETS = ['breastmnist', 'dermamnist', 'pneumoniamnist', 'organcmnist']

NEW_TARGET_DATASETS = ['pathmnist', 'tissuemnist', 'octmnist', 'chestmnist']

MODELS = ['resnet18', 'resnet50']

EXPERIMENTS_LIST = []

for src_dataset in SOURCE_DATASETS:
    for src_model in MODELS:
        
        # --- FILTRO DE SEGURANÇA ---
        # Não tens o pipeline do Pneumonia ResNet50, por isso saltamos.
        if src_dataset == 'pneumoniamnist' and src_model == 'resnet50':
            continue
            
        src_folder = f"VAL_output_csv_VAL_optimize_do_{src_dataset}_{src_model}"
        
        for tgt_dataset in NEW_TARGET_DATASETS:
            
           
            for tgt_model in MODELS:
                
                test_id = f"TestUnseen_Src_{src_dataset[:6]}{src_model[-2:]}_Tgt_{tgt_dataset[:4]}{tgt_model[-2:]}"
                
                EXPERIMENTS_LIST.append({
                    "test_name": test_id,
                    "source_folder": src_folder,
                    "source_dataset": src_dataset,      # De onde vem a aug
                    "target_dataset": tgt_dataset,      # O dataset novo
                    "target_model": tgt_model,          # O modelo no dataset novo
                    "resolution": 28,                   
                    "seeds": [1]                        
                })

print(f"Total de Experiências 'Unseen' geradas: {len(EXPERIMENTS_LIST)}")
'''
BATCH_SIZE = 64
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_MAIN_DIR = "FINAL_TRANSFER_RESULTS"


def get_transforms(resolution):
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    
    if resolution == 224:
        before_augs = [A.Resize(224, 224, interpolation=cv2.INTER_CUBIC)]
    else:
        before_augs = [] 

    after_augs = [
        A.Resize(resolution, resolution, interpolation=cv2.INTER_CUBIC), 
        A.Normalize(mean=mean, std=std), 
        ToTensorV2()
    ]
    return before_augs, after_augs

def find_global_champion(folder, dataset_filter):
    print(f"   > Searching for champion in: '{folder}'...")
    search_pattern = os.path.join(folder, f"*{dataset_filter}*.csv")
    files = glob.glob(search_pattern)
    
    if not files:
        raise FileNotFoundError(f"No files found in {folder}")
        
    global_best_fit = -1.0
    champion_genotype_str = None
    champion_source = ""
    
    for f in files:
        try:
            try: df = pd.read_csv(f, sep=';')
            except: df = pd.read_csv(f, sep=',')
            
            if 'best_fitness' not in df.columns: continue
            
            idx_max = df['best_fitness'].idxmax()
            row = df.loc[idx_max]
            
            if row['best_fitness'] > global_best_fit:
                global_best_fit = row['best_fitness']
                champion_genotype_str = row['best_individual']
                champion_source = f
        except: pass
            
    if champion_genotype_str is None:
        raise ValueError("Could not extract a valid genotype.")

    return champion_genotype_str, os.path.basename(champion_source)

def run_single_experiment(exp_config):
    test_name = exp_config['test_name']
    src_folder = exp_config['source_folder']
    src_data = exp_config['source_dataset']
    tgt_data = exp_config['target_dataset']
    tgt_model = exp_config['target_model']
    res = exp_config['resolution']
    seeds_list = exp_config['seeds']

    print(f"\n>>> Start Experience: {test_name}")
    print(f"    Origin: {src_data} | Destiny: {tgt_data} ({tgt_model})")
    
    current_out_dir = os.path.join(OUTPUT_MAIN_DIR, test_name)
    os.makedirs(current_out_dir, exist_ok=True)
    matrix_folder = os.path.join(current_out_dir, "matrices")
    os.makedirs(matrix_folder, exist_ok=True)
    output_csv = os.path.join(current_out_dir, f"Results_{test_name}.csv")

    try:
        policy_str, source_file = find_global_champion(src_folder, src_data)
        champion_individual_list = ast.literal_eval(policy_str)
        print(f"    Política carregada de: {source_file}")
    except Exception as e:
        print(f"    [ERRORR] loaded policy: {e}")
        return 

    config = config_base.config.copy()
    
    config['DATA_FLAG'] = tgt_data
    config['cache_folder'] = f"cache_{tgt_data}_torch"
    
   
    tgt_info = data_loader.medmnist_dataset.INFO[tgt_data]
    n_classes = len(tgt_info['label'])
    config['num_classes'] = n_classes
    config['num_classes_downstream'] = n_classes
    if 'num_classes_pretext' not in config: config['num_classes_pretext'] = 2
    
    print(f"    [Config] Dataset: {tgt_data} | Classes: {n_classes} | Cache: {config['cache_folder']}")

    config['device'] = DEVICE
    config['epochs'] = EPOCHS
    config['dataset_transforms'] = lambda: get_transforms(res)

    try:
        trainset, valset, testset = data_loader.load_dataset_simple(champion_individual_list, config)
    except Exception as e:
        print(f"!!! ERROR loaded dataset: {e}")
        return

    results_data = []
    vals_auc, vals_acc, tests_auc, tests_acc = [], [], [], []

    for seed in seeds_list:
        print(f"    > Executando Seed {seed}...")
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed(seed)
        
        config['confusion_matrix_config'] = {
            'print_confusion_matrix': True,
            'confusion_matrix_folder': matrix_folder,
            'confusion_matrix_downstream_file': f"cm_seed_{seed}.txt",
            'num_classes_downstream': n_classes, 
            'confusion_matrix_pretext_file': "dummy.txt",
            'num_classes_pretext': 2
        }

        trainloader, valloader, testloader = data_loader.create_data_loaders_simple(trainset, valset, testset, config)

        if tgt_model == 'resnet50':
            model = net_models_torch.TrainResNet50Simple(num_classes_downstream=n_classes)
        else:
            model = net_models_torch.TrainResNet18Simple(num_classes_downstream=n_classes)
        
      
        start_time = time.time()
        sl_eval.train_sl(model, trainloader, config)
        duration = time.time() - start_time

        config_val = config.copy()
        config_val['confusion_matrix_config']['print_confusion_matrix'] = False
        
        val_acc, val_auc, _, _, _ = sl_eval.run_inference(model, valloader, DEVICE, config_val['confusion_matrix_config'], config_val)
        test_acc, test_auc, _, _, _ = sl_eval.run_inference(model, testloader, DEVICE, config['confusion_matrix_config'], config)

        vals_auc.append(val_auc); vals_acc.append(val_acc)
        tests_auc.append(test_auc); tests_acc.append(test_acc)

        results_data.append({
            "Experiment": test_name, "Seed": seed,
            "Val AUC": val_auc, "Val ACC": val_acc,
            "Test AUC": test_auc, "Test ACC": test_acc,
            "Time": round(duration, 2)
        })
        
        del model
        torch.cuda.empty_cache()
        gc.collect()

    results_data.append({
        "Experiment": "SUMMARY", "Seed": "AVG ± STD",
        "Val AUC": f"{np.mean(vals_auc):.4f} ± {np.std(vals_auc):.4f}",
        "Val ACC": f"{np.mean(vals_acc):.4f} ± {np.std(vals_acc):.4f}",
        "Test AUC": f"{np.mean(tests_auc):.4f} ± {np.std(tests_auc):.4f}",
        "Test ACC": f"{np.mean(tests_acc):.4f} ± {np.std(tests_acc):.4f}",
        "Time": "-"
    })

    pd.DataFrame(results_data).to_csv(output_csv, index=False, sep=';')
    print(f"    [DONE] Average Test AUC: {np.mean(tests_auc):.4f}")

if __name__ == "__main__":
    
    print(f"Starting batch of {len(EXPERIMENTS_LIST)} experiments...")
    
    for i, experiment in enumerate(EXPERIMENTS_LIST):
        try:
            run_single_experiment(experiment)
        except Exception as e:
            print(f"!!! Failure in experiment {experiment['test_name']}: {e}")
            import traceback
            traceback.print_exc()
            
    print("\n--- ALL EXPERIMENTS COMPLETED ---")