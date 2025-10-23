import ast
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_FOLDER = 'output_csv_optimize_downstream'

EXPERIMENTS = ['cifar10_optimize_downstream']

SEEDS = [0,5]

# BASELINE = 0.728
BASELINE = 0.671

ONLY_BESTS = True

ONLY_TOP_INDIVIDUAL = False

USE_DA_NAMES = True

START_AT_GEN = None

STOP_AT_GEN = None

da_pr_usage = {}

for experiment in EXPERIMENTS:
    for seed in SEEDS:
        df = pd.read_csv(os.path.join(BASE_FOLDER, f'{experiment}_{seed}.csv'), sep=';')

        if STOP_AT_GEN is not None:
            df = df[df['generation'] <= STOP_AT_GEN]

        if START_AT_GEN is not None:
            df = df[df['generation'] >= START_AT_GEN]

        if ONLY_TOP_INDIVIDUAL:
            # get the row which contains the max value for the best_fitness column
            row = df.loc[df['best_fitness'].idxmax()]
            ind = ast.literal_eval(row['best_individual'])
            for chromo in ind:
                da_pr_usage_dict = da_pr_usage.get(chromo[0], {})
                for i in range(5):
                    da_pr_usage_dict[i] = da_pr_usage_dict.get(i, []) + [chromo[1][i]]
                da_pr_usage[chromo[0]] = da_pr_usage_dict
        else:
            if ONLY_BESTS:
                for fit, ind in zip(df['best_fitness'], df['best_individual']):
                    if BASELINE is not None and fit <= BASELINE:
                        continue
                    ind = ast.literal_eval(ind)
                    for chromo in ind:
                        da_pr_usage_dict = da_pr_usage.get(chromo[0], {})
                        for i in range(5):
                            da_pr_usage_dict[i] = da_pr_usage_dict.get(i, []) + [chromo[1][i]]
                        da_pr_usage[chromo[0]] = da_pr_usage_dict
            else:
                for pop in df['population']:
                    pop = ast.literal_eval(pop)
                    for ind in pop:
                        fit = ind[1]
                        if BASELINE is not None and fit <= BASELINE:
                            continue
                        for chromo in ind[0]:
                            da_pr_usage_dict = da_pr_usage.get(chromo[0], {})
                            for i in range(5):
                                da_pr_usage_dict[i] = da_pr_usage_dict.get(i, []) + [chromo[1][i]]
                            da_pr_usage[chromo[0]] = da_pr_usage_dict


print(da_pr_usage)

if USE_DA_NAMES:
    import data_augmentation_torch
    def get_da_name(da):
        return str(data_augmentation_torch.da_funcs[da](0,0,0,0,0)).split('(')[0]
    
    def da_names_no_duplicates():
        da_idx = []
        da_names = []
        for k in da_pr_usage.keys():
            da_name = get_da_name(k)
            da_idx.append(k)
            da_names.append(da_name)
        
        non_duplicate_da_names = {}

        for idx, da_name in zip(da_idx, da_names):
            if da_names.count(da_name) > 1:
                non_duplicate_da_names[idx] = f'{da_name}_{idx}'
            else:
                non_duplicate_da_names[idx] = da_name
        return non_duplicate_da_names
    
    da_names_dict = da_names_no_duplicates()
    print(da_names_dict)
        
        
for da_function, param_dict in da_pr_usage.items():
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for param_idx in range(5):
        ax = axes[param_idx]

        if len(np.unique(param_dict[param_idx])) == 1:
            # plot a vertical line if all values are the same
            ax.axvline(x=param_dict[param_idx][0], color='red', linewidth=2)
        else:
            sns.kdeplot(param_dict[param_idx], ax=ax, fill=True)
        
        ax.set_title(f'Parameter {param_idx}')
        ax.set_xlim(0, 1)
        ax.set_xlabel('Value')
        ax.set_ylabel('Probability Density')

    # Hide the 6th subplot (bottom-right) since there are only 5 parameters
    axes[5].axis('off')

    fig.suptitle(f'Distribution of Parameters for {da_function if not USE_DA_NAMES else da_names_dict[da_function]}', fontsize=16)
    plt.tight_layout()
    plt.show()