def train_and_evaluate_individual_preprocessed_dataset(individual, config):
    
    dataset_vars = config['load_dataset_func'](config)

    loaders = config['data_loader_func'](*dataset_vars, individual, config)

    downstream_acc, pretext_acc, hist = config['model_evaluate_func'](*loaders, config)

    print("Fitness (Accuracy):", downstream_acc)

    return downstream_acc, pretext_acc, hist

        

def train_and_evaluate_individual(individual, config):
    
    dataset_vars = config['load_dataset_func'](individual, config)

    loaders = config['data_loader_func'](*dataset_vars, config)

    downstream_acc, pretext_acc, hist = config['model_evaluate_func'](*loaders, config)

    print("Fitness (Accuracy):", downstream_acc)

    return downstream_acc, pretext_acc, hist
