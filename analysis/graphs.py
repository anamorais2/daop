import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PRINT_TOTAL_TIME_AND_EXIT = True

PLOT_MAX_GEN_IND = False
PLOT_OVERALL_MAX = True
PLOT_NEW_MAX_SCARCE = False

PLOT_AVG_OVERALL_BESTS = True

PLOT_AVG_GEN_BESTS = False
PLOT_AVG_GEN_AVGS = False
# PLOT_AVG_GEN_BESTS = True
# PLOT_AVG_GEN_AVGS = True

# MAX_FITNESS_LINE = 'None'
MAX_FITNESS_LINE = '--'
PLOT_BASELINES = True

# BASE_FOLDER = "output_csv_optimize_pretext"
# BASE_FOLDER = "output_csv_optimize_downstream_20_20"
BASE_FOLDER = "output_csv_optimize_simultaneous_100"
# BASE_FOLDER = "output_csv_optimize_simultaneous_20_400"

PRINT_AVG_OVERALL_BESTS = True
PRINT_OVERALL_BESTS = True

FONT_SIZE = 28

# experiments=['cifar10_optimize_pretext']
# experiments=['cifar10_optimize_downstream']
experiments = ['cifar10_optimize_simultaneous_100']
# experiments = ['cifar10_optimize_simultaneous_20_400']

# baselines = {"NA": 0.58, "RN": 0.671, "RN100": 0.728}
baselines = {"NA": 0.58, "RN100": 0.728}
# baselines = {"NA": 0.58, "RN": 0.671}

# checkpoints = {200: 'EG200'}
checkpoints = {}

# seeds = range(10)
# seeds = [i for i in range(10) if i not in [5,9]]
seeds = [0,1,2]

cut_at_gen = None
cut_at_gen = (1,200)

overall_min_gen = None
overall_max_gen = None

if cut_at_gen:
    overall_min_gen = cut_at_gen[0]
    overall_max_gen = cut_at_gen[1]

generations = range(cut_at_gen[0], cut_at_gen[1]+1) if cut_at_gen else None


y_range = "auto"
x_range = "auto"

y_range = [0.55, 0.75]
# y_range = [0.55, 0.7]
overall_max_fit = 0
overall_min_fit = 1
overall_bests = []

for exp in experiments:
    avgs = []
    bests = []
    run_new_bests = {}
    max_fit = 0
    min_fit = 1

    total_time = 0
    for seed in seeds:
        df = pd.read_csv(os.path.join(BASE_FOLDER, f'{exp}_{seed}.csv'), sep=';')
        avgs.append(df['avg_fitness'])
        bests.append(df['best_fitness'])
        max_fit = max(max_fit, df['best_fitness'].max())
        min_fit = min(min_fit, df['avg_fitness'].min())
        if generations is None:
            generations = df['generation']
            if overall_min_gen is None:
                overall_min_gen = df['generation'].min()
            else:
                overall_min_gen = min(overall_min_gen, df['generation'].min())
            if overall_max_gen is None:
                overall_max_gen = df['generation'].max()
            else:
                overall_max_gen = max(overall_max_gen, df['generation'].max())

        total_time += df['total_time'][overall_min_gen-1:overall_max_gen].sum()

        if len(overall_bests) == 0:
            overall_bests = df['best_fitness'][overall_min_gen-1:overall_max_gen].to_list()
        else:
            for i in range(overall_min_gen-1, overall_max_gen):
                overall_bests[i] = max(overall_bests[i], df['best_fitness'][i])

        if PLOT_AVG_OVERALL_BESTS:
            run_new_bests[seed] = []
            for i in range(overall_min_gen-1, overall_max_gen):
                if len(run_new_bests[seed]) == 0 or df['best_fitness'][i] > run_new_bests[seed][-1]:
                    run_new_bests[seed].append(df['best_fitness'][i])
                else:
                    run_new_bests[seed].append(run_new_bests[seed][-1])

    if PRINT_TOTAL_TIME_AND_EXIT:
        print(f'Total time: {total_time/(24*60*60)}')
        sys.exit(0)

    print(len(generations), overall_min_gen, overall_max_gen)
    if cut_at_gen:
        avgs = [avg[cut_at_gen[0]-1:cut_at_gen[1]] for avg in avgs]
        bests = [best[cut_at_gen[0]-1:cut_at_gen[1]] for best in bests]
        # run_new_bests = {seed: run_new_bests[seed][cut_at_gen[0]-1:cut_at_gen[1]] for seed in run_new_bests}
    print(len(generations), len(avgs[-1]), len(bests[-1]))
    print(len(avgs), len(bests))
    # print(np.max(avgs[0]), np.max(avgs[1]), np.max(avgs[2]))
    # print(np.max(bests[0]), np.max(bests[1]), np.max(bests[2]))
    avgs_avg = np.array(avgs).mean(axis=0)
    bests_avg = np.array(bests).mean(axis=0)
    label_best = 'BGIF' # 'Best Population Individual Fitness ' + exp if len(experiments) > 1 else 'Best Population Individual Fitness'
    label_avg = 'AGF'# 'Average Population Fitness ' + exp if len(experiments) > 1 else 'Average Population Fitness'
    # label_best = label_best + f' avg {len(seeds)} runs' if exp in seeds and len(seeds) > 1 else label_best
    # label_avg = label_avg + f' avg {len(seeds)} runs' if exp in seeds and len(seeds) > 1 else label_avg
    if PLOT_AVG_GEN_BESTS:
        plt.plot(generations, bests_avg, label=label_best, color="tab:red", marker='x')
        # for i in range(len(avgs_avg)):
        #     print(f'Avg bests at gen {generations[i]}: {bests_avg[i]} [{','.join([str(bests[j][i]) for j in range(len(bests))])}]')
    if PLOT_AVG_GEN_AVGS:
        plt.plot(generations, avgs_avg, label=label_avg, color="tab:orange", marker='o')

    if PLOT_AVG_OVERALL_BESTS:
        run_new_bests_arr = np.array([run_new_bests[seed] for seed in seeds])
        print(run_new_bests_arr.shape)
        mean_new_bests = run_new_bests_arr.mean(axis=0)
        std_new_bests = run_new_bests_arr.std(axis=0)
        print(mean_new_bests.shape)

        if PRINT_AVG_OVERALL_BESTS:
            for i in range(len(mean_new_bests)):
                print(f'Avg overall bests at gen {generations[i]}: {mean_new_bests[i]} +/- {std_new_bests[i]}, [{','.join([str(x) for x in sorted(run_new_bests_arr[:,i], reverse=True)])}]')
        
        # lbl = ('Avg ' if len(seeds) > 1 else '') + 'Run Overall Bests' + (f' {exp}' if len(experiments) > 1 else '')
        plt.plot(generations, mean_new_bests, label='ROB', color='tab:blue', marker='x', linestyle=MAX_FITNESS_LINE)

    overall_max_fit = max(overall_max_fit, max_fit)
    overall_min_fit = min(overall_min_fit, min_fit)

if PLOT_BASELINES:
    for label, b in baselines.items():
        baseline = [b]*len(generations)
        plt.plot(generations, baseline, label=f'Baseline {label}', linewidth=2)


if PLOT_MAX_GEN_IND:
    plt.plot(generations, overall_bests[:len(generations)], label='MGIF', color='green', marker='*', linestyle=MAX_FITNESS_LINE)
if PLOT_OVERALL_MAX:
    new_best_fits = []
    for i in range(len(generations)):
        if len(new_best_fits) == 0 or overall_bests[i] > new_best_fits[-1]:
            new_best_fits.append(overall_bests[i])
        else:
            new_best_fits.append(new_best_fits[-1])

    if PRINT_OVERALL_BESTS:
        for i in range(len(new_best_fits)):
            print(f'Overall max at gen {generations[i]}: {new_best_fits[i]}')
    plt.plot(generations, new_best_fits, label='OMF', color='green', marker='*', linestyle=MAX_FITNESS_LINE)
if PLOT_NEW_MAX_SCARCE:
    new_best_fits = []
    new_best_gens = []
    for i in range(len(generations)):
        if len(new_best_fits) == 0 or overall_bests[i] > new_best_fits[-1]:
            new_best_fits.append(overall_bests[i])
            new_best_gens.append(generations[i])
    plt.plot(new_best_gens, new_best_fits, label='NMF', color='green', marker='*', markersize=10, linestyle=MAX_FITNESS_LINE)


if checkpoints is not None:
    for gen, label in checkpoints.items():
        if gen in generations:
            # vertical line at checkpoint
            plt.axvline(x=gen, color='red', linewidth=2, label=f'Checkpoint {label}')

        
if y_range == "auto":
    fit_range = overall_max_fit - overall_min_fit
    plt.ylim(overall_min_fit - 0.05*fit_range, overall_max_fit + 0.05*fit_range)
elif isinstance(y_range, list):
    plt.ylim(y_range)

if x_range == "auto":
    gen_range = overall_max_gen - overall_min_gen
    plt.xlim(overall_min_gen - 0.05*gen_range, overall_max_gen + 0.05*gen_range)
elif isinstance(x_range, list):
    plt.xlim(x_range)

plt.xticks(fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.xlabel('Generation', fontsize=FONT_SIZE)
plt.ylabel('Fitness', fontsize=FONT_SIZE)
plt.title('Evolution of Fitness\n' + (f"Average {len(seeds)} runs" if len(seeds) > 1 else f"Seed {seeds[0]}"), fontsize=FONT_SIZE)
plt.legend(loc='lower right', fontsize=FONT_SIZE)
plt.grid(True)
plt.show()