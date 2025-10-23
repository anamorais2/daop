import dataset.data_processing_no_ssl as data_processing_no_ssl
import no_ssl

import configs.config_base as config_base

config = config_base.config
config['fix_pretext_da'] = []
config['base_experiment_name'] = "optimize_sl"
config['experiment_name'] = config['base_experiment_name']
config['output_csv_folder'] = "output_csv" + "_" + config['base_experiment_name']
config['load_dataset_func'] = data_processing_no_ssl.load_dataset
config['data_loader_func'] = data_processing_no_ssl.create_data_loaders
config['model_evaluate_func'] = no_ssl.evaluate_no_ssl