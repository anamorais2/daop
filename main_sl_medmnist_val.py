import os
import sys
import pandas as pd
import ast
import analysis.plot_roc as plot_roc
import configs.config_base_val as config_base
import EA_medmnist_val as EA


def completed_run(config):
    file_path = os.path.join(config['output_csv_folder'], f'{config["dataset"]}_{config["experiment_name"]}_{config["seed"]}.csv')

  
    if not os.path.exists(file_path):
        return False


    df = pd.read_csv(file_path, sep=';')
    max_gen = df['generation'].iloc[-1]
    stop_gen = config['stop_gen'] + 1 if config['extended_isolated_run'] else config['stop_gen']
    if max_gen >= stop_gen:
        del df
        return True
    

    # Get the last generation and parent
    config['start_gen'] = max_gen + 1
    start_parent_str = df['best_individual'].iloc[-1]

    try:
        config['start_parent'] = ast.literal_eval(start_parent_str)
        print(f"Recovered parent: {config['start_parent']}")
    except:
        print(f"Failed to parse parent {start_parent_str}")
        config['start_parent'] = None

    # Get the best individuals
    df_top_n = df.nlargest(config['best_n'], 'best_fitness')
    for individual, fitness in zip(df_top_n['best_individual'], df_top_n['best_fitness']):
        try:
            config['best_individuals'].append([ast.literal_eval(individual), fitness, None, None, None])
            print(f"Recovered top 5 individual: {config['best_individuals'][-1]}")
        except:
            print(f"Failed to parse top 5 individual {individual}")

    config['best_individuals'].sort(key=lambda x: x[1], reverse=True)


    # Check if the state file exists
    state_file = f'state_{config["dataset"]}_{config["experiment_name"]}_{config["seed"]}.pickle'
    if os.path.exists(os.path.join(config['state_folder'], state_file)):
        config['state_file'] = state_file

    del df
    return False


def reset_config(config):
    config['start_gen'] = 1
    config['start_parent'] = None
    config['best_individuals'] = []
    config['state_file'] = None
    config['epochs'] = config['base_epochs']
    config['pretext_epochs'] = config['base_pretext_epochs']
    config['downstream_epochs'] = config['base_downstream_epochs']
    config['pretrained_pretext_model'] = config['base_pretrained_pretext_model']
    
    
def generate_final_roc_plot(config):
   
    file_path = os.path.join(config['output_csv_folder'], f'{config["dataset"]}_{config["experiment_name"]}_{config["seed"]}.csv')
    
    if not os.path.exists(file_path):
        print(f"ERROR: Results file {file_path} does not exist. Cannot generate final ROC plot.")
        return

    df = pd.read_csv(file_path, sep=';')
    
    # Select the row with the highest recorded best_auc
    best_row = df.loc[df['auc_test'].idxmax()]
    
    best_genotype_str = best_row['best_individual']
    best_auc = best_row['auc_test']

    try:
        best_genotype = ast.literal_eval(best_genotype_str)
    except Exception as e:
        print(f"ERROR: Failed to parse genotype string for plotting. {e}")
        return
    
    print(f"\n--- STARTING FINAL ROC PLOT (Seed {config['seed']}) ---")
    print(f"Best DA policy (genotype) found. AUC: {best_auc:.4f}")
    
    # 2. FINAL RE-EVALUATION (CALL TRAIN/EVALUATION FUNCTION)
    # Re-evaluate the best genotype to obtain raw curve data (FPR/TPR).
    # We call the evaluation function stored in config (typically train_with_DA.py).
    #
    # NOTE: This will retrain the model (e.g. for base_epochs). To avoid retraining,
    # modify the evaluation function to support loading a saved model / test-only mode.
    
    print("Re-evaluating best individual to obtain curve data (FPR/TPR)...")
    _, _, history = config['individual_evaluation_func'](best_genotype, config)
    
    # 3. Extract raw data and plot
    fpr_final = history.get('fpr') 
    tpr_final = history.get('tpr')
    
    if fpr_final is not None and tpr_final is not None:
        file_name_prefix = f"{config['dataset']}_{config['experiment_name']}_{config['seed']}_FINAL_ROC"
        output_folder = os.path.join(config['output_csv_folder'], 'final_plots')
        
        # Call plotting utility (plot_roc.py)
        plot_roc.plot_roc_curve_and_save(fpr_final, tpr_final, best_auc, file_name_prefix, output_folder)
        print("Final ROC plot completed.")
    else:
        print("Warning: Could not obtain FPR/TPR data. Check the return value of the evaluation function.")


skip_runs = [
    # seed
    # 0,
    # 1,
]

# skip_until_mutation = 1
skip_until_run = None
    
if __name__ == "__main__":
    config = config_base.config

    try:
        if len(sys.argv) > 1:
            config['seeds'] = [int(i) for i in sys.argv[1].split(",")]
    except:
        print("Failed to parse seeds from command line arguments")

    for seed in config['seeds']:
        config['seed'] = seed
        reset_config(config)


        if completed_run(config) or seed in skip_runs or (skip_until_run is not None and seed != skip_until_run):
            print(f"Skipping seed {seed}")
            continue
        print(f"Running seed {seed}")
        EA.ea(config)
        
        generate_final_roc_plot(config)


    if config['delete_cache']:
        import shutil
        shutil.rmtree(config['cache_folder'])
