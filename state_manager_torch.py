import torch
import numpy as np
import random
import pickle
import os

def save_state(config):
    if not os.path.exists(config['state_folder']):
        os.makedirs(config['state_folder'])

    state_file_path = os.path.join(config['state_folder'], f'state_{config["dataset"]}_{config["experiment_name"]}_{config["seed"]}.pickle')

    with open(state_file_path, 'wb') as f:
        pickle.dump((random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state()), f)

def load_state(config):
    if config['state_file'] is not None:
        with open(os.path.join(config['state_folder'], config['state_file']), 'rb') as f:
            random_state, np_state, torch_state, torch_cuda_state = pickle.load(f)
            random.setstate(random_state)
            np.random.set_state(np_state)
            torch.set_rng_state(torch_state)
            torch.cuda.set_rng_state(torch_cuda_state)
        print("Loaded state from file")
        config['state_file'] = None
    else: 
        random.seed(config['seed'])
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        torch.cuda.manual_seed(config['seed'])
