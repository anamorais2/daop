import pandas as pd
import numpy as np
import glob
import os

ROOT_DIR = "."

datasets_map = {
    "BreastMNIST": "breastmnist",
    "DermaMNIST": "dermamnist",
    "PneumoniaMNIST": "pneumoniamnist",
    "OrganCMNIST": "organcmnist"
}


def get_time_stats(dataset_folder_key, model_key):
    folder_pattern = os.path.join(ROOT_DIR, f"*{dataset_folder_key}*{model_key}*")
    folders = glob.glob(folder_pattern)
    
    if not folders:
        return None
    
    all_times = []
    
    for folder in folders:
        csv_files = glob.glob(os.path.join(folder, "*.csv"))
        
        for f in csv_files:
            try:
                df = pd.read_csv(f, sep=';')
                total_hours = df['total_time'].sum() / 3600
                all_times.append(total_hours)
            except:
                pass

    if not all_times:
        return None

    return {
        "mean": np.mean(all_times),
        "std": np.std(all_times, ddof=1),
        "count": len(all_times)
    }

print("="*75)
print(f"{'Dataset':<15} | {'ResNet-18 (h)':<25} | {'ResNet-50 (h)':<25}")
print("-" * 75)

latex_rows = []

for pretty_name, folder_key in datasets_map.items():
    row_str = f"{pretty_name:<15}"
    
    stats_r18 = get_time_stats(folder_key, "resnet18")
    stats_r50 = get_time_stats(folder_key, "resnet50")
    
    if stats_r18:
        val_r18 = f"{stats_r18['mean']:.3f} ± {stats_r18['std']:.3f}"
        latex_r18 = f"{stats_r18['mean']:.3f}h $\\pm$ {stats_r18['std']:.3f}h"
        row_str += f" | {val_r18:<25}"
    else:
        row_str += f" | {'N/A':<25}"
        latex_r18 = "-"

    if stats_r50:
        val_r50 = f"{stats_r50['mean']:.3f} ± {stats_r50['std']:.3f}"
        latex_r50 = f"{stats_r50['mean']:.3f}h $\\pm$ {stats_r50['std']:.3f}h"
        row_str += f" | {val_r50:<25}"
    else:
        row_str += f" | {'N/A':<25}"
        latex_r50 = "-"
        
    print(row_str)
    
    latex_rows.append(f"\\textbf{{{pretty_name}}} & {latex_r18} & {latex_r50} \\\\")

print("="*75)
print("\n--- LaTeX Code ---")
for row in latex_rows:
    print(row)