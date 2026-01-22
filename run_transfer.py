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


import configs.config_base_val_transfer as config_base
import net_models_torch
import sl_evaluation_medmnist_val as sl_eval
import dataset.data_processing_medmnist_val_transfer as data_loader
import DA.data_augmentation_albumentations as da_lib 

CSV_FOLDER = "VAL_output_csv_VAL_optimize_do_dermamnist_resnet18"        
DATASET_NAME = 'dermamnist' 
RESOLUTION_HIGH = 224            
BATCH_SIZE = 64
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRANSFER_SEEDS = [1, 2, 3, 4, 5]       
MODEL_TYPE = "resnet18"          

OUTPUT_FILENAME = f"Transfer_Results_{DATASET_NAME}_{MODEL_TYPE}_{RESOLUTION_HIGH}.csv"
MATRIX_FOLDER = f"transfer_results_matrices_{DATASET_NAME}_{MODEL_TYPE}"


def high_res_transforms():
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    
    before_augs = [
        A.Resize(RESOLUTION_HIGH, RESOLUTION_HIGH, interpolation=cv2.INTER_CUBIC)
    ]
    
    # Resize FINAL:
    # Se a política campeã fizer um Crop para 28px, isto força a voltar a 224px 
    # para garantir que todas as imagens do batch têm o mesmo tamanho.
    after_augs = [
        A.Resize(RESOLUTION_HIGH, RESOLUTION_HIGH, interpolation=cv2.INTER_CUBIC), # <--- NOVO
        A.Normalize(mean=mean, std=std), 
        ToTensorV2()
    ]
    
    return before_augs, after_augs

def find_global_champion(folder, dataset_name):
    print(f"Look for the champion in '{folder}'...")
    search_pattern = os.path.join(folder, f"{dataset_name}_*.csv")
    files = glob.glob(search_pattern)
    
    if not files:
        print(f"File does not exist for {dataset_name}!")
        sys.exit(1)
        
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
                
        except Exception as e:
            pass
        
    print(f"Origin: {os.path.basename(champion_source)}")
    print(f"Val Acc (Low-Res Origin): {global_best_fit:.4f}")
    

    return champion_genotype_str, os.path.basename(champion_source)

def run_and_save():
    policy_str, source_file = find_global_champion(CSV_FOLDER, DATASET_NAME)
    

    champion_individual_list = ast.literal_eval(policy_str)

    if not os.path.exists(MATRIX_FOLDER):
        os.makedirs(MATRIX_FOLDER, exist_ok=True)

    print("\n" + "="*50)
    print(f"Transfer initiated ({DATASET_NAME} -> {RESOLUTION_HIGH}px)")
    print("="*50)

    results_data = []
    
    config = config_base.config.copy()
    config['device'] = DEVICE
    config['epochs'] = EPOCHS
    
    config['dataset_transforms'] = high_res_transforms

 
    print("Loading datasets using 'load_dataset_simple'...")
    trainset, valset, testset = data_loader.load_dataset_simple(champion_individual_list, config)
    
    num_classes = config['num_classes']
    config['num_classes_downstream'] = num_classes
    
    if 'num_classes_pretext' not in config:
        config['num_classes_pretext'] = 2

    test_auc_scores = []
    test_acc_scores = []
    val_auc_scores = []
    val_acc_scores = []

    for seed in TRANSFER_SEEDS:
        print(f"\nSeed: {seed}...")
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        matrix_filename = f"confusion_matrix_seed_{seed}.txt"
        config['confusion_matrix_config'] = {
            'print_confusion_matrix': True,
            'confusion_matrix_folder': MATRIX_FOLDER,
            'confusion_matrix_downstream_file': matrix_filename,
            'num_classes_downstream': num_classes,
            'confusion_matrix_pretext_file': "dummy_pretext.txt",
            'num_classes_pretext': config['num_classes_pretext']
        }
   
        trainloader, valloader, testloader = data_loader.create_data_loaders_simple(trainset, valset, testset, config)
        
        if MODEL_TYPE == 'resnet50':
            model = net_models_torch.TrainResNet50Simple(num_classes_downstream=config['num_classes_downstream'])
        else:
            model = net_models_torch.TrainResNet18Simple(num_classes_downstream=config['num_classes_downstream'])
        
        start_time = time.time()
        sl_eval.train_sl(model, trainloader, config)
        duration = time.time() - start_time
        
        config_val = config.copy()
        config_val['confusion_matrix_config'] = config['confusion_matrix_config'].copy()
        config_val['confusion_matrix_config']['print_confusion_matrix'] = False
        
        val_acc, val_auc, _, _, _ = sl_eval.run_inference(model, valloader, DEVICE, config_val['confusion_matrix_config'], config_val)
        

        test_acc, test_auc, _, _, _ = sl_eval.run_inference(model, testloader, DEVICE, config['confusion_matrix_config'], config)
        
        print(f"Seed {seed} | Val AUC: {val_auc:.4f} | Test AUC: {test_auc:.4f}")
        
        val_auc_scores.append(val_auc)
        val_acc_scores.append(val_acc)
        test_auc_scores.append(test_auc)
        test_acc_scores.append(test_acc)
        
        results_data.append({
            "Run Type": "Individual Seed",
            "Dataset": DATASET_NAME,
            "Resolution": RESOLUTION_HIGH,
            "Seed": seed,
            "Val AUC": val_auc,
            "Val ACC": val_acc,
            "Test AUC": test_auc,
            "Test ACC": test_acc,
            "Policy Source": source_file,
            "Total Time": str(duration),
            "Matrix File": matrix_filename
        })


    mean_val_auc = np.mean(val_auc_scores)
    std_val_auc = np.std(val_auc_scores)
    mean_val_acc = np.mean(val_acc_scores)
    std_val_acc = np.std(val_acc_scores)
    
    mean_test_auc = np.mean(test_auc_scores)
    std_test_auc = np.std(test_auc_scores)
    mean_test_acc = np.mean(test_acc_scores)
    std_test_acc = np.std(test_acc_scores)

    results_data.append({
        "Run Type": "FINAL SUMMARY",
        "Dataset": DATASET_NAME,
        "Resolution": RESOLUTION_HIGH,
        "Seed": "MEAN ± STD",
        "Val AUC": f"{mean_val_auc:.4f} ± {std_val_auc:.4f}",
        "Val ACC": f"{mean_val_acc:.4f} ± {std_val_acc:.4f}",
        "Test AUC": f"{mean_test_auc:.4f} ± {std_test_auc:.4f}",
        "Test ACC": f"{mean_test_acc:.4f} ± {std_test_acc:.4f}",
        "Policy Source": "-",
        "Total Time": "-",
        "Matrix File": "-"
    })

    df_results = pd.DataFrame(results_data)
    cols = ["Run Type", "Dataset", "Resolution", "Seed", "Val AUC", "Val ACC", "Test AUC", "Test ACC", "Policy Source", "Total Time", "Matrix File"]
    df_results = df_results[[c for c in cols if c in df_results.columns]]
    
    df_results.to_csv(OUTPUT_FILENAME, index=False, sep=';')
    
    print("\n" + "="*50)
    print(f"Results saved in : {OUTPUT_FILENAME}")
    print(f"Val AUC: {mean_val_auc:.4f} ± {std_val_auc:.4f}")
    print(f"Test AUC: {mean_test_auc:.4f} ± {std_test_auc:.4f}")
    print("="*50)

if __name__ == "__main__":
    data_loader.DATA_FLAG = DATASET_NAME 
    data_loader.INFO = data_loader.medmnist_dataset.INFO[DATASET_NAME]
    data_loader.DataClass = getattr(data_loader.medmnist_dataset, data_loader.INFO['python_class'])
    run_and_save()