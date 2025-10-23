import ast
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_FOLDER = 'output_csv_optimize_simultaneous'
# BASE_FOLDER = 'output_csv_optimize_simultaneous_100'
# BASE_FOLDER = 'output_csv_optimize_simultaneous_20_400'

EXPERIMENTS = ['cifar10_optimize_simultaneous']
# EXPERIMENTS = ['cifar10_optimize_simultaneous_100']
# EXPERIMENTS = ['cifar10_optimize_simultaneous_20_400']

SEEDS = range(10)
# SEEDS = [0,1,2]

# BASELINE = 0.728
# BASELINE = 0.671
BASELINE = None

ONLY_BESTS = True

ONLY_TOP_INDIVIDUAL = True

USE_DA_NAMES = True

IGNORE_DUPLICATES = True

START_AT_GEN = None

# STOP_AT_GEN = 200
STOP_AT_GEN = None

# Y_RANGE = (0,10)
Y_RANGE = None

FONT_SIZE = 20

da_count_pretext = {}
da_count_downstream = {}

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

            curr_ind = []
            for chromo in ind[0]:
                if chromo[0] in curr_ind and IGNORE_DUPLICATES:
                    continue
                curr_ind.append(chromo[0])
                da_count_pretext[chromo[0]] = da_count_pretext.get(chromo[0], 0) + 1
            for chromo in ind[1]:
                if chromo[0] in curr_ind and IGNORE_DUPLICATES:
                    continue
                curr_ind.append(chromo[0])
                da_count_downstream[chromo[0]] = da_count_downstream.get(chromo[0], 0) + 1
        else:
            if ONLY_BESTS:
                for fit, ind in zip(df['best_fitness'], df['best_individual']):
                    if BASELINE is not None and fit <= BASELINE:
                        continue

                    ind = ast.literal_eval(ind)
                    
                    curr_ind = []
                    for chromo in ind[0]:
                        if chromo[0] in curr_ind and IGNORE_DUPLICATES:
                            continue
                        curr_ind.append(chromo[0])
                        da_count_pretext[chromo[0]] = da_count_pretext.get(chromo[0], 0) + 1
                    for chromo in ind[1]:
                        if chromo[0] in curr_ind and IGNORE_DUPLICATES:
                            continue
                        curr_ind.append(chromo[0])
                        da_count_downstream[chromo[0]] = da_count_downstream.get(chromo[0], 0) + 1
            else:
                for pop in df['population']:
                    pop = ast.literal_eval(pop)
                    for ind in pop:
                        fit = ind[1]
                        if BASELINE is not None and fit <= BASELINE:
                            continue

                        curr_ind = []
                        for chromo in ind[0][0]:
                            if chromo[0] in curr_ind and IGNORE_DUPLICATES:
                                continue
                            curr_ind.append(chromo[0])
                            da_count_pretext[chromo[0]] = da_count_pretext.get(chromo[0], 0) + 1
                        for chromo in ind[0][1]:
                            if chromo[0] in curr_ind and IGNORE_DUPLICATES:
                                continue
                            curr_ind.append(chromo[0])
                            da_count_downstream[chromo[0]] = da_count_downstream.get(chromo[0], 0) + 1
        
        
# plot the da_count distribution in a sorted histogram

phase_names = ['pretext', 'downstream']

for phase_name, da_count in zip(phase_names, [da_count_pretext, da_count_downstream]):

    da_count = {k: v for k, v in sorted(da_count.items(), key=lambda item: item[1], reverse=True)}

    print(da_count)

    if USE_DA_NAMES:
        import data_augmentation_torch

        SPECIAL_DA_NAMES = {
            "Compose": "Crop",
        }

        FINAL_DA_NAMES = {
            "Illumination_35": "Illumination_Linear",
            "Illumination_36": "Illumination_Corner",
            "Illumination_37": "Illumination_Gaussian",
            "Affine_4": "Translate",
            "Affine_5": "Shear",
            "Sharpen_15": "Sharpen_Kernel",
            "Sharpen_16": "Sharpen_Gaussian"
        }
        
        def get_da_name(da):
            temp_da_name = str(data_augmentation_torch.da_funcs[da](0,0,0,0,0)).split('(')[0]
            if temp_da_name in SPECIAL_DA_NAMES:
                return SPECIAL_DA_NAMES[temp_da_name]
            return temp_da_name
        
        def da_names_no_duplicates():
            da_idx = []
            da_names = []
            for k in da_count.keys():
                da_name = get_da_name(k)
                da_idx.append(k)
                da_names.append(da_name)
            
            non_duplicate_da_names = []

            for idx, da_name in zip(da_idx, da_names):
                if da_names.count(da_name) > 1:
                    non_duplicate_da_names.append(f'{da_name}_{idx}')
                else:
                    non_duplicate_da_names.append(da_name)

            for idx, da_name in enumerate(non_duplicate_da_names):
                if da_name in FINAL_DA_NAMES:
                    non_duplicate_da_names[idx] = FINAL_DA_NAMES[da_name]

            return non_duplicate_da_names
        
        # print([get_da_name(k) for k in da_count.keys()])
        # plt.bar([get_da_name(k) for k in da_count.keys()], da_count.values())

        print(da_names_no_duplicates())
        plt.bar(da_names_no_duplicates(), da_count.values())
        
        plt.xticks(rotation=60, fontsize=FONT_SIZE)
        plt.yticks(ticks=range(0, max(da_count.values()) + 1), fontsize=FONT_SIZE)
        plt.subplots_adjust(bottom=0.3)
    else:
        plt.bar([str(k) for k in da_count.keys()], da_count.values())
    for i, v in enumerate(da_count.values()):
        plt.text(i, v, str(v), ha='center', va='bottom', fontsize=FONT_SIZE)
    plt.xlabel('DA', fontsize=FONT_SIZE)
    plt.ylabel('Frequency', fontsize=FONT_SIZE)

    if Y_RANGE is not None:
        plt.ylim(Y_RANGE)

    plt.yticks(fontsize=FONT_SIZE)
    plt.title('Distribution of DAs in ' + phase_name, fontsize=FONT_SIZE)
    plt.show()

