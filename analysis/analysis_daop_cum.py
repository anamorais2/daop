import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
import glob
import traceback

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def plot_evolution(df, output_file, title_text):
    plt.figure(figsize=(10, 6))

    generation = df['generation']
    avg_fit = df['avg_fitness_val']
    best_fit = df['best_so_far'] 
    
    std_fit = df['std_fitness_val']
    upper = avg_fit + std_fit
    lower = avg_fit - std_fit



    plt.plot(generation, best_fit, color='#2980b9', label='Best Individual', zorder=3)

    plt.plot(generation, avg_fit, color='#c0392b', linestyle='--', label='Population Avg.', zorder=2)

    plt.fill_between(generation, lower, upper, color='#c0392b', alpha=0.15, label='Std. Deviation', zorder=1)
    
    plt.title(title_text, fontsize=14)
    plt.xlabel('Generations')
    plt.ylabel('Fitness (Accuracy)') 
    
    plt.legend(loc='lower right', frameon=True, framealpha=0.9)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight') 
    plt.close()
    print(f" > Individual Plot Saved: {os.path.basename(output_file)}")

def plot_aggregated(all_data_list, output_file, title_text):
    if not all_data_list:
        print(" ! Warning: No data available for aggregated plot.")
        return

    best_seed_name = "Unknown"
    highest_val = -1.0

    aligned_series = []
    
    for item in all_data_list:
        df = item['df']
        seed_name = item['seed']
        
        s = df.set_index('generation')['best_so_far']
        s = s[~s.index.duplicated(keep='first')]
        aligned_series.append(s)
        
        current_max = s.max()
        if current_max > highest_val:
            highest_val = current_max
            best_seed_name = seed_name

    combined_df = pd.concat(aligned_series, axis=1)
    
    mean_curve = combined_df.mean(axis=1)
    std_curve = combined_df.std(axis=1)
    max_curve = combined_df.max(axis=1) 
    
    generations = combined_df.index 

    plt.figure(figsize=(10, 6))

    plt.plot(generations, max_curve, color='#27ae60', linestyle='--', linewidth=1.5, 
             label=f'Max Fitness (Best: Seed {best_seed_name})', alpha=0.9)

    plt.plot(generations, mean_curve, color='#2980b9', linewidth=2.5, 
             label='Average Fitness (All Seeds)')

    upper = mean_curve + std_curve
    lower = mean_curve - std_curve
    plt.fill_between(generations, lower, upper, color='#2980b9', alpha=0.15, 
                     label='Standard Deviation')

    plt.title(title_text, fontsize=14)
    plt.xlabel('Generations')
    plt.ylabel('Fitness (Accuracy)')

    
    plt.legend(loc='lower right', frameon=True, framealpha=0.9)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" >> AGGREGATED PLOT SAVED: {os.path.basename(output_file)}")


if __name__ == "__main__":

    # --- INPUT SETTINGS ---
    INPUT_FOLDER = "/home/arya/Documents/carolina/daop/VAL_output_csv_VAL_optimize_do_pneumoniamnist_resnet50"
    OUTPUT_FOLDER = "pneumoniamnist_resnet50_graphs"

    MAX_GENERATIONS = 200 
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    csv_files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))
    print(f"Found {len(csv_files)} CSV files in '{INPUT_FOLDER}'...")

    all_data_list = [] 

    for csv_file in csv_files:
        try:
            base_name = os.path.basename(csv_file)
            name_parts = base_name.split('_')
            
            if len(name_parts) < 5:
                seed_info = base_name.replace('.csv', '')
            else:
                seed_info = name_parts[-1].replace('.csv', '')

            df = pd.read_csv(csv_file, sep=';')
            
            cols_to_use = ['avg_fitness_val', 'std_fitness_val', 'best_fitness']
            for col in cols_to_use:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
         
            df = df.drop_duplicates(subset='generation', keep='first')
            
            df = df[df['generation'] <= MAX_GENERATIONS]
            
            df = df.sort_values(by='generation')

            df['best_so_far'] = df['best_fitness'].cummax()
            
            all_data_list.append({'seed': seed_info, 'df': df})

            output_filename = os.path.join(OUTPUT_FOLDER, f"evolution_{seed_info}_dermamnist.png")
            chart_title = f"Seed {seed_info}: DermaMNIST (ResNet-50)"
            
            plot_evolution(df, output_filename, chart_title)

        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            traceback.print_exc()

    if all_data_list:
        print("\nGenerating aggregated graph...")
        agg_output = os.path.join(OUTPUT_FOLDER, "ALL_dermaMNIST_ResNet50.png")
        agg_title = f"Optimization Progress: DermaMNIST (ResNet-50) (Avg. {len(all_data_list)} Seeds)"
        
        plot_aggregated(all_data_list, agg_output, agg_title)

    print("\nProcessing Complete!")