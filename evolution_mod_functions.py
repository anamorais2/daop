from numpy import copy
import state_manager_torch

def extended_gens(config, past_gen=False):
    config['epochs'] = config['extended_epochs']
    print(f"Extended gens mod: extended epochs to {config['epochs']}")

    if config['extended_pretrained_pretext_model'] is not None:
        config['pretrained_pretext_model'] = config['extended_pretrained_pretext_model']
        print(f"Extended gens mod: pretrained pretext model changed to extended version {config['pretrained_pretext_model']}")

    if not past_gen:
        config['start_population'] = [[individual[0], individual[1], None, None, None] for individual in config['best_individuals']]
        config['best_individuals'] = []
        print(f"Extended gens mod:  Resetting population to {config['start_population']}")
        
def extended_gens(config, past_gen=False):
    config['epochs'] = config['extended_epochs']


    if not past_gen:
        if config['state_file'] is not None:
            try:
                state_manager_torch.load_state(os.path.join(config['state_folder'], config['state_file']))
                print(f"Loaded state from {config['state_file']}")
            except Exception as e:
                print(f"Error loading state: {e}")

        population = [copy.deepcopy(individual) for individual in config['best_individuals']]
        
        i = 0
        while len(population) < config['population_size']:
            population.append(config['mutation'](config['best_individuals'][i], config))
            i = (i + 1) % len(config['best_individuals'])
            
        config['start_population'] = population
        best_gen_individual = max(population, key=lambda x: x[1])
        
        return population, best_gen_individual

    return None, None