import pandas as pd
import matplotlib.pyplot as plt
import os
import sys



def analyze_single_file(file_path, output_filename, dataset_name, network_name):

    try:
        df = pd.read_csv(file_path, sep=';')
    except Exception as e:
        print(f"  ERROR reading {file_path}: {e}. Skipping this file.")
        return

    # Ensure fitness columns are numeric
    numeric_cols = ['generation', 'avg_fitness', 'std_fitness', 'best_fitness']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            print(f"  WARNING: Column '{col}' not found in {file_path}.")
            return 
            
    df = df.dropna(subset=numeric_cols)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(18, 7))
    
    csv_base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    title = f'DAOP Analysis: {dataset_name} on {network_name} (File: {csv_base_name})'
    fig.suptitle(title, fontsize=16)

    generation = df['generation']
    avg_fitness = df['avg_fitness']
    std_fitness = df['std_fitness']
    
    upper_band = avg_fitness + std_fitness
    lower_band = avg_fitness - std_fitness

    ax1.plot(generation, avg_fitness, color='#FF5733', label='Mean Fitness (Avg. Fitness)')
    ax1.fill_between(generation, lower_band, upper_band, color='#FF5733', alpha=0.3, label='Mean Fitness ± Std. Deviation')
    
    ax1.set_title('Population Convergence')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness (Accuracy)')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.plot(df['generation'], df['best_fitness'], color='green', marker='o', markersize=3, linestyle='-', label='Best Fitness')
    
    ax2.set_title('Optimization Progress (Best Fitness)')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Best Fitness (Accuracy)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plt.savefig(output_filename)
    plt.close(fig) 



def analyze_multi_seed_comparison(csv_files_map, output_filename, dataset_name, network_name):
    
    plt.figure(figsize=(14, 9))
    ax = plt.gca() 

    for seed_label, file_path in csv_files_map.items():
        
        try:
            df = pd.read_csv(file_path, sep=';')
        except Exception as e:
            print(f"  WARNING: Error reading {file_path}: {e}. Skipping.")
            continue

        numeric_cols = ['generation', 'best_fitness']
        for col in df.columns:
            if col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=numeric_cols)

        ax.plot(df['generation'], df['best_fitness'], marker='o', 
                markersize=2, linestyle='-', alpha=0.7, 
                label=f'Seed {seed_label}')

    
    title = f'DAOP Analysis: Best Fitness Evolution ({dataset_name} on {network_name})'
    ax.set_title(title)
    
    ax.set_xlabel('Generation')
    ax.set_ylabel('Best Fitness (Accuracy)')
    ax.legend(loc='lower right', title='Seeds')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()


if __name__ == "__main__":
    
    DATASET_NAME = "BreastMNIST"
    NETWORK_NAME = "ResNet50"
    
    files_to_compare = {
        '1': 'breastmnist_DA_optimize_do_breastmnist_DA_resnet50_1.csv',
        '2': 'breastmnist_DA_optimize_do_breastmnist_DA_resnet50_2.csv',
        '3': 'breastmnist_DA_optimize_do_breastmnist_DA_resnet50_3.csv',
        '4': 'breastmnist_DA_optimize_do_breastmnist_DA_resnet50_4.csv',
        '5': 'breastmnist_DA_optimize_do_breastmnist_DA_resnet50_5.csv',
        '6': 'breastmnist_DA_optimize_do_breastmnist_DA_resnet50_6.csv',
        '7': 'breastmnist_DA_optimize_do_breastmnist_DA_resnet50_7.csv',
        '8': 'breastmnist_DA_optimize_do_breastmnist_DA_resnet50_8.csv',
        '9': 'breastmnist_DA_optimize_do_breastmnist_DA_resnet50_9.csv',
        '10': 'breastmnist_DA_optimize_do_breastmnist_DA_resnet50_10.csv'
    }

    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_script_dir) 
    
    output_plot_dir = os.path.join(parent_dir, 'analysis_results')
    os.makedirs(output_plot_dir, exist_ok=True)

 
    csv_input_dir = os.path.join(parent_dir, 'output_csv_optimize_do_breastmnist_DA_resnet50')

    print(f"Looking for CSV files in: {csv_input_dir}")
    existing_files_map = {}
    for seed, relative_path in files_to_compare.items():
        csv_full_path = os.path.join(csv_input_dir, relative_path)
        
        if os.path.exists(csv_full_path):
            existing_files_map[seed] = csv_full_path
        else:
            print(f"Note: Input CSV '{relative_path}' not found in {csv_input_dir}. Skipping.")
    
    if not existing_files_map:
        print(f"ERROR: No input CSV files found in {csv_input_dir}. Analysis aborted.")
        sys.exit(1)

    print(f"Found {len(existing_files_map)} files. Starting analysis...")

    print("\nRunning Analysis 1: Multi-Seed Comparison...")
    comparison_output_path = os.path.join(output_plot_dir, f'daop_comparison_{DATASET_NAME}_{NETWORK_NAME}.png')
    
    analyze_multi_seed_comparison(existing_files_map, comparison_output_path, DATASET_NAME, NETWORK_NAME)
    print(f"Comparison chart saved to: {comparison_output_path}")

    print("\nRunning Analysis 2: Individual Seed Details...")
    for seed, path in existing_files_map.items():
        print(f"Analyzing Seed {seed} ({os.path.basename(path)})...")
        
        base_name = os.path.splitext(os.path.basename(path))[0]
        individual_output_path = os.path.join(output_plot_dir, f"{base_name}_analysis.png")
        
        analyze_single_file(path, individual_output_path, DATASET_NAME, NETWORK_NAME)
    
    print(f"\nAnalysis complete. All charts saved to '{output_plot_dir}' folder.")