import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import scikit_posthocs as sp

experiments=[['cifar10_mut_exp_0.33_0.33_0.66', 'cifar10_mut_exp_0.5_0.66_0.33', 'cifar10_mut_exp_0.33_0.33_0.33'], 
             ['cifar10_mut_exp_0.33_0.33_0.66', 'cifar10_mut_exp_0.5_0.66_0.33'], 
             ['cifar10_mut_exp_0.33_0.33_0.66', 'cifar10_mut_exp_0.33_0.33_0.33'], 
             ['cifar10_mut_exp_0.33_0.33_0.33', 'cifar10_mut_exp_0.5_0.66_0.33'],
             ]

unique_experiments = list(set([e for experiment in experiments for e in experiment]))

folders = {
    'mut_exp_killed_16_0.33_0.33_0.33': 'results_killed',
    'mut_exp_killed_0.33_0.33_0.5': 'results_killed',
    'mut_exp_killed_0.33_0.33_0.33': 'results_killed',
    'mut_exp_10k_0.33_0.33_0.33': 'results_killed',
    'mut_exp_shuffle_0.33_0.33_0.33': 'results_shuffle',
}


seeds = {e: [0, 29, 42] for e in unique_experiments}
seeds['mut_exp_0.33_0.5_0.5'] = [0, 29]

cut_at_gen = {
    'mut_exp_10k_0.33_0.33_0.33': (1,45)
}


for e in range(len(experiments)):
    experiment = experiments[e]

    best_fitnesses = []

    

    for exp_config in experiment:
        best_fit = []
        for seed in (seeds[exp_config] if exp_config in seeds else [0]):
            if exp_config in folders:
                df = pd.read_csv(os.path.join('output_csv', folders[exp_config], f'{exp_config}_{seed}.csv'), sep=';')
            else:
                df = pd.read_csv(os.path.join('output_csv', f'{exp_config}_{seed}.csv'), sep=';')
            
            best_fit.append(df['best_fitness'])
        best_fit = np.array(best_fit)

        best_fitnesses.append(np.mean(best_fit, axis=0))

    if len(experiment) == 2:
        u_stat, p_value = stats.mannwhitneyu(best_fitnesses[0], best_fitnesses[1], alternative='greater')

        print(f"Results of the Mann-Whitney U test: u_stat={u_stat}, p_value={p_value}")
    else:
        stat, p_value = stats.friedmanchisquare(*best_fitnesses)

        print(f"Results of the Friedman test: stat={stat}, p_value={p_value}")

        if p_value < 0.05:
            posthoc = sp.posthoc_mannwhitney(best_fitnesses)
            print(f"Posthoc results: {posthoc}")



