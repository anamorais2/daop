import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Plot styling configuration
FONT_SIZE = 14
plt.rcParams.update({'font.size': FONT_SIZE})

def plot_evolution_rob_omf(folder_path, experiment_name="Experiment", baseline=None, output_filename="evolution_plot.png"):
    """
    Generates the evolution plot with ROB (Run Overall Bests) and OMF (Overall Max Fitness).
    Optimized to read only specific columns from CSVs to avoid parsing errors.
    """
    
    # 1. Find all CSV files in the folder
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    if not all_files:
        print(f"ERROR: No CSV files found in: {folder_path}")
        return

    print(f"Processing {len(all_files)} seeds from: {folder_path}")

    # List to store best_fitness series from each file
    data_frames = []

    for filename in all_files:
        try:
            # OPTIMIZATION: Read only necessary columns to avoid parsing errors with large dicts
            df = pd.read_csv(
                filename, 
                sep=';', 
                usecols=['generation', 'best_fitness'], 
                on_bad_lines='skip'
            )
            
            # Ensure data is numeric (coercing errors to NaN)
            df['generation'] = pd.to_numeric(df['generation'], errors='coerce')
            df['best_fitness'] = pd.to_numeric(df['best_fitness'], errors='coerce')
            
            # Remove invalid rows
            df = df.dropna()
            
            if not df.empty:
                # Sort and set index
                df = df.sort_values('generation')
                df = df.set_index('generation')
                
                # Keep only best_fitness column
                data_frames.append(df['best_fitness'])
            else:
                print(f"Warning: File {os.path.basename(filename)} has no valid numeric data.")
                
        except Exception as e:
            # Often happens if the header is missing or columns are named differently
            print(f"Error reading {os.path.basename(filename)}: {e}")

    if not data_frames:
        print("No valid data loaded. Check CSV headers.")
        return

    # 2. Consolidate Data
    # Create a single DataFrame (Rows=Generations, Cols=Seeds)
    df_all = pd.concat(data_frames, axis=1)
    
    # Fill missing values (forward fill)
    df_all = df_all.ffill() 

    # 3. Calculate Metrics
    
    # --- ROB (Run Overall Bests) ---
    # Cumulative max for each seed ("best so far")
    df_cummax = df_all.cummax()
    # Average and Standard Deviation across seeds
    rob_mean = df_cummax.mean(axis=1)
    rob_std = df_cummax.std(axis=1)

    # --- OMF (Overall Max Fitness) ---
    # Max absolute value across ALL seeds per generation
    max_per_gen = df_all.max(axis=1)
    # Cumulative max of that series (World Record)
    omf_curve = max_per_gen.cummax()

 
    plt.figure(figsize=(10, 6))
    
    generations = rob_mean.index
    num_seeds = df_all.shape[1]

    # Plot OMF (Green dashed line - Potential Max)
    plt.plot(generations, omf_curve, label='OMF (Overall Max Fitness)', 
             color='green', linestyle='--', linewidth=2, alpha=0.9)

    # Plot ROB (Blue solid line - Average Expected Performance)
    plt.plot(generations, rob_mean, label='ROB (Run Overall Bests)', 
             color='blue', linewidth=2)
    
    # Standard Deviation Shadow
    plt.fill_between(generations, rob_mean - rob_std, rob_mean + rob_std, 
                     color='blue', alpha=0.15, label='Standard Deviation (ROB)')

    # Baseline Line (if provided)
    if baseline is not None:
        plt.axhline(y=baseline, color='red', linestyle='-.', linewidth=2, 
                    label=f'Baseline ({baseline})', alpha=0.8)

    # Chart labels
    plt.title(f'Evolution of Fitness - {experiment_name}\n(Average {num_seeds} runs)', fontsize=16)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    # Save Plot
    output_path = os.path.join(folder_path, output_filename)
    plt.savefig(output_path, dpi=300)
    print(f"\nEvolution plot saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        folder = sys.argv[1]
        # Optional: Experiment Name
        exp_name = sys.argv[2] if len(sys.argv) > 2 else "DAOP Experiment"
        # Optional: Baseline Value
        baseline_val = float(sys.argv[3]) if len(sys.argv) > 3 else None
        
        plot_evolution_rob_omf(folder, exp_name, baseline_val)
        
    else:
        print("Usage: python graphs.py <path_to_csv_folder> [Experiment_Name] [Baseline_Value]")
        print("Example: python graphs.py output_csv_breastmnist_r18 'BreastMNIST ResNet18' 0.86")