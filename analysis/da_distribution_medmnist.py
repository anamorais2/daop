import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import albumentations as A


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.dirname(current_dir)) 

try:
    import DA.data_augmentation_albumentations as da_module
except ImportError:
    try:
        sys.path.append(os.path.join(os.path.dirname(current_dir), 'DA'))
        #import data_augmentation_albumentations as da_module
    except ImportError:
        print("CRITICAL ERROR: Could not find 'data_augmentation_albumentations.py'.")
        sys.exit(1)


def get_da_names():
    """Executes lambda functions with dummy values to get specific class names/modes."""
    dummy_size = (28, 28) 
    funcs = da_module.da_funcs_probs(0.0, 1.0, dummy_size)
    names = []
    
    for func in funcs:
        try:
          
            tf = func(0.5, 0.5, 0.5, 0.5, 0.5)
            name = tf.__class__.__name__
            
            if name == 'Compose':
                sub_names = [t.__class__.__name__ for t in tf.transforms]
                name = "+".join(sub_names)
    
            elif name == 'Illumination':
                mode = getattr(tf, 'mode', 'unknown')
                name = f"Illumination ({mode})"
            
         
            elif name == 'RandomSnow':
                method = getattr(tf, 'method', 'unknown')
                name = f"RandomSnow ({method})"

            names.append(name)
        except:
            names.append(f"Unknown_{len(names)}")
            
    return names

DA_NAMES = get_da_names()
print(f"Loaded {len(DA_NAMES)} augmentation types (Detailed).")

def analyze_da_distribution(folder_path, experiment_name="Experiment"):
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    if not all_files:
        print(f"ERROR: No CSV files found in: {folder_path}")
        return

    print(f"Analyzing DA distribution in {len(all_files)} seeds...")

    da_counts = {i: 0 for i in range(len(DA_NAMES))}
    total_individuals = 0

    for filename in all_files:
        try:
            df = pd.read_csv(filename, sep=';')
            
            # Get best individual
            if 'best_fitness' in df.columns and 'best_individual' in df.columns:
                best_row = df.loc[df['best_fitness'].idxmax()]
                genotype_str = best_row['best_individual']
                
                try:
                    genotype = ast.literal_eval(genotype_str)
                except:
                    print(f"Error parsing genotype in {os.path.basename(filename)}")
                    continue

                augs_list = []
                
                # Robust logic to handle SL vs DO structure
                if isinstance(genotype, list) and len(genotype) > 0:
                    
                    if len(genotype) == 2 and isinstance(genotype[0], list) and isinstance(genotype[1], list):
                        
                        if len(genotype[0]) == 0:
                            augs_list = genotype[1]
                            
                        elif len(genotype[0]) > 0 and isinstance(genotype[0][0], int):
                            augs_list = genotype
                        
                        else:
                            augs_list = genotype[1]

                    elif isinstance(genotype[0], list):
                        augs_list = genotype
                        
                    elif isinstance(genotype[0], int):
                        augs_list = [genotype]

                # Count occurrences
                for aug in augs_list:
                    if isinstance(aug, list) and len(aug) > 0:
                        idx = aug[0]
                        if idx < len(DA_NAMES):
                            da_counts[idx] += 1
                
                total_individuals += 1

        except Exception as e:
            print(f"Error processing {os.path.basename(filename)}: {e}")


    active_augs = {k: v for k, v in da_counts.items() if v > 0}
    
    if not active_augs:
        print("No augmentations found.")
        return

    sorted_augs = sorted(active_augs.items(), key=lambda item: item[1], reverse=True)
    labels = [DA_NAMES[k] for k, v in sorted_augs]
    values = [v for k, v in sorted_augs]

    plt.figure(figsize=(14, 8))
    bars = plt.bar(labels, values, color='skyblue', edgecolor='black')
    
    plt.xlabel('Data Augmentation Technique', fontsize=14)
    plt.ylabel('Frequency (Count)', fontsize=14)
    plt.title(f'Most Selected Augmentations - {experiment_name}\n(Across {total_individuals} seeds)', fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, int(yval), ha='center', va='bottom')

    plt.tight_layout()
    
    output_file = os.path.join(folder_path, f'{experiment_name}_da_distribution.png')
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to: {output_file}")
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python da_distribution.py <csv_folder_path> [Experiment_Name]")
    else:
        folder = sys.argv[1]
        name = sys.argv[2] if len(sys.argv) > 2 else "DA Distribution"
        analyze_da_distribution(folder, name)