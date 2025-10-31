import copy
import os
import sys
import time
import numpy as np
import random
import gc
from pympler.tracker import SummaryTracker
import sl_evaluation_medmnist as sl_evaluation_medmnist



import pandas as pd
import os
import numpy as np # Necessário para np.mean/np.std

def write_gen_stats(config, gen_stats, population, best_individual, best_auc):
  
    # Usamos best_individual[1] (a fitness) e best_individual[0] (o genótipo)
    avg_fitness = np.mean([individual[1] for individual in population])
    std_fitness = np.std([individual[1] for individual in population])
    
    # O objeto best_individual que entra na função JÁ É o melhor indivíduo.
    best_fitness_val = best_individual[1]
    best_genotype_str = str(best_individual[0]) # Genótipo (lista de aumentos)

    generation = gen_stats['generation']
    total_time = gen_stats['total_time']
    
    stats_dict = {
        'generation': generation,
        'avg_fitness': avg_fitness,
        'std_fitness': std_fitness,
        'best_fitness': best_fitness_val,
        'best_auc': best_auc, # Valor escalar
        'total_time': total_time,
        'best_individual': best_genotype_str,
        
    }

    df = pd.DataFrame([stats_dict])
 
    if not os.path.exists(config['output_csv_folder']):
        os.makedirs(config['output_csv_folder'])

    file_path = os.path.join(config['output_csv_folder'], f'{config["dataset"]}_{config["experiment_name"]}_{config["seed"]}.csv')

    write_header = not os.path.exists(file_path)
    
    try:
        df.to_csv(file_path, mode='a', header=write_header, index=False, sep=';')
        
    except Exception as e:
        print(f"ERRO: Falha ao escrever estatísticas no ficheiro {file_path}: {e}")
        
        # Tentar escrever no ficheiro de backup em caso de falha de I/O
        file_path_backup = os.path.join(config['output_csv_folder'], f'{config["dataset"]}_{config["experiment_name"]}_{config["seed"]}_backup.csv')
        print(f"Tentando escrever no ficheiro de backup: {file_path_backup}")
        
        write_header_backup = not os.path.exists(file_path_backup)
        try:
            df.to_csv(file_path_backup, mode='a', header=write_header_backup, index=False, sep=';')
            print("Escrita no backup bem-sucedida.")
        except Exception as e_backup:
            print(f"ERRO CRÍTICO: Falha ao escrever no ficheiro de backup {file_path_backup}: {e_backup}")
            print(f"Dados perdidos da Geração {generation}: {stats_dict}")


def create_individual(config):
    genotype = [
        config["fix_pretext_da"] if config['fix_pretext_da'] is not None else None,     # 0 - pretext
        config["fix_downstream_da"] if config['fix_downstream_da'] is not None else None,   # 1 - downstream
    ]

    if config['fix_pretext_da'] is None or config['fix_downstream_da'] is None:
        if config['evolution_type'] == 'same':
            print("Same evolution type creation")
            chromosomes = [config['create_chromosome']() for _ in range(random.randint(1, config['max_chromosomes']))]
            if config['fix_pretext_da'] is None:
                genotype[0] = chromosomes   # pretext chromosomes
            if config['fix_downstream_da'] is None:
                genotype[1] = chromosomes   # downstream chromosomes
        elif config['evolution_type'] == 'simultaneous':
            print("Simultaneous evolution type creation")
            if config['fix_pretext_da'] is None:
                genotype[0] = [config['create_chromosome']() for _ in range(random.randint(1, config['max_chromosomes']))]  # pretext chromosomes
            if config['fix_downstream_da'] is None:
                genotype[1] = [config['create_chromosome']() for _ in range(random.randint(1, config['max_chromosomes']))]  # downstream chromosomes
        else:
            raise ValueError(f"Unknown evolution type: {config['evolution_type']}")



    individual = [
        genotype,  # chromosomes for pretext and downstream
        None,  # fitness
        None,  # pretext accuracy
        None,  # training time
        None,  # training history
    ]
    return individual

def ea(config):

    if config['check_memory_leaks']:
        tracker = SummaryTracker()
        tracker.print_diff()

    if config['load_state']:
        config['load_state'](config)

    if config['start_parent'] is not None:
        best_gen_individual = [config['start_parent'], None, None, None, None]
        print(f"Starting from parent: {best_gen_individual[0]}")
        config['start_parent'] = None
    else:
        best_gen_individual = create_individual(config)

    for gen, evolution_mod in config['evolution_mods'].items():
        if gen >= config['start_gen']:
            break

        evolution_mod(config, past_gen=True)

    # main evolution loop
    for gen in range(config['start_gen'], config['stop_gen']+1):
        if config['max_generations_per_run'] is not None:
            config['current_run_generations'] += 1
            if config['current_run_generations'] > config['max_generations_per_run']:
                print("Max generations reached")
                sys.exit()

        print("Generation: ", gen)

        if config['every_gen_state_reset']:
            config['every_gen_state_reset'](config)

        if config['evolution_mods'].get(gen) is not None:
            print(f"Running evolution mod for generation {gen}")
            config['evolution_mods'][gen](config)

        # new generation population
        if config['start_population'] is None:
            population = [copy.deepcopy(best_gen_individual)]
            for _ in range(config['population_size']-1):
                genotype = [
                    config["fix_pretext_da"] if config['fix_pretext_da'] is not None else None,     # 0 - pretext
                    config["fix_downstream_da"] if config['fix_downstream_da'] is not None else None,   # 1 - downstream
                ]

                if config['fix_pretext_da'] is None or config['fix_downstream_da'] is None:
                    if config['evolution_type'] == 'same':
                        # print("Same evolution type mutation")
                        mutation_phase = 0 if config['fix_pretext_da'] is None else 1   # pretext or downstream chromosome mutation if fixed DAs are used
                        chromosomes = config['mutation'](best_gen_individual[0][mutation_phase], config)
                        if config['fix_pretext_da'] is None:
                            genotype[0] = chromosomes   # mutated pretext chromosomes
                        if config['fix_downstream_da'] is None:
                            genotype[1] = chromosomes   # mutated downstream chromosomes
                    elif config['evolution_type'] == 'simultaneous':
                        # print("Simultaneous evolution type mutation")
                        if config['fix_pretext_da'] is None:
                            genotype[0] = config['mutation'](best_gen_individual[0][0], config)  # mutated pretext chromosomes
                        if config['fix_downstream_da'] is None:
                            genotype[1] = config['mutation'](best_gen_individual[0][1], config)  # mutated downstream chromosomes
                    else:
                        raise ValueError(f"Unknown evolution type: {config['evolution_type']}")


                offspring = [
                    genotype,    # mutated chromosomes
                    None,  # reset fitness 
                    None,  # reset pretext accuracy
                    None,  # reset training time
                    None,  # reset training history
                ]
                population.append(offspring)

            print("Parent:", population[0][:-1])
        else:
            population = config['start_population']
            config['start_population'] = None

        for individual in population:
            print(individual[0])

        # get fitness of each individual
        for i in range(config['population_size']):
            individual = population[i]
            print(f"\nTraining individual {i+1} for {config['epochs']} epochs: {individual[0]}")
            if individual[1] is None or config['recalculate_best']:
                
                start_time = time.perf_counter()
                individual[1], individual[2], individual[4] = config['individual_evaluation_func'](individual[0], config)
                end_time = time.perf_counter()

                # individual[1], individual[2], individual[4] = [i, i, [i]]
                print("Epochs: ", config['epochs'])

                individual[3] = end_time - start_time

                gc.collect()

                print(f"Training time: {end_time - start_time:.2f} seconds")


        # best individual
        best_gen_individual = copy.deepcopy(max(population, key=lambda x: x[1]))
        
        write_gen_stats(config, gen, population, best_gen_individual,best_auc = best_gen_individual[4].get('sl_auc', -1))

        if config['save_state']:
            config['save_state'](config)

        # print("Best individual:", config['best_individual'])

        # check if new top n was found
        if len(config['best_individuals']) < config['best_n'] or best_gen_individual[1] > config['best_individuals'][-1][1]:
            if len(config['best_individuals']) >= config['best_n']:
                del config['best_individuals'][-1]
            config['best_individuals'].append(best_gen_individual)
            config['best_individuals'].sort(key=lambda x: x[1], reverse=True)
            print(f"New Top {config['best_n']} individual:", best_gen_individual[:-1])

            # list top n individuals
            print(f"Updated Top {config['best_n']} individuals:")
            for i, individual in enumerate(config['best_individuals']):
                print(f"\tTop {i+1}/{config['best_n']} individual: {individual[:-1]}")



        for i in range(config['population_size']-1, -1, -1):
            if population[i] != best_gen_individual:
                for j in range(len(population[i])-1, -1, -1):
                    del population[i][j]
                del population[i]

        # print("Population size", len(population))

        del population

        if config['check_memory_leaks']:
            tracker.print_diff()


    if config['extended_isolated_run']:

        print(f"Extending best individuals isolatedly for {config['extended_epochs']} epochs")

        gen = config['stop_gen']

        config['epochs'] = config['extended_epochs']
        config['pretext_epochs'] = config['extended_pretext_epochs']
        config['downstream_epochs'] = config['extended_downstream_epochs']

        population = [[individual[0], individual[1], None, None, None] for individual in config['best_individuals']]

        for i, individual in enumerate(population):
            start_time = time.perf_counter()

            individual[1], individual[2], individual[4] = config['individual_evaluation_func'](individual[0], config)

            end_time = time.perf_counter()

            individual[3] = end_time - start_time

            print(f"Top {i+1}/{len(population)} individual isolated extended run accuracy: {individual[1]:.2f}")
            print(f"Top {i+1}/{len(population)} individual isolated extended run training time: {individual[3]:.2f} seconds")

        write_gen_stats(config, gen+1, population=population, best_individual=max(population, key=lambda x: x[1]), best_auc=max(population, key=lambda x: x[4]).get('sl_auc', -1))
