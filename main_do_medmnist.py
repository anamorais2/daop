import os
import sys

import pandas as pd
import ast

import analysis.plot_roc as plot_roc

import configs.config_base_medmnist as config_base
import EA_medmnist as EA


def completed_run(config):
    file_path = os.path.join(config['output_csv_folder'], f'{config["dataset"]}_{config["experiment_name"]}_{config["seed"]}.csv')

    # Check if the file exists
    if not os.path.exists(file_path):
        return False 
                          

    # Check if the stop_gen has been reached
    # print(file_path)
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
        print(f"ERRO: O ficheiro de resultados {file_path} não existe. Impossível gerar a plotagem ROC final.")
        return

    df = pd.read_csv(file_path, sep=';')
    
    # 1. Encontrar a linha com a melhor fitness (AUC ou Accuracy)
    # Usamos 'best_auc' que foi guardado pelo EA_simple.py
    best_row = df.loc[df['best_auc'].idxmax()]
    
    best_genotype_str = best_row['best_individual']
    best_auc = best_row['best_auc']

    try:
        best_genotype = ast.literal_eval(best_genotype_str)
    except Exception as e:
        print(f"ERRO: Falha ao interpretar o genótipo para plotagem. {e}")
        return
    
    print(f"\n--- INICIANDO PLOTAGEM ROC FINAL (Seed {config['seed']}) ---")
    print(f"Melhor Política DA (Genótipo) Encontrada. AUC: {best_auc:.4f}")
    
    # 2. REAVALIAÇÃO FINAL (CHAMADA À FUNÇÃO DE TREINO/TESTE)
    # Reavaliamos o melhor genótipo para obter os dados brutos da curva (FPR/TPR)
    # Usamos a função de avaliação do config (que aponta para train_with_DA.py)
    
    # ATENÇÃO: Isto irá retreinar o modelo (ex: 20 épocas).
    # Se quiser evitar o retreino, teria de modificar o train_with_DA.py
    # para aceitar uma flag 'test_only=True' e carregar o modelo guardado.
    
    # Por agora, vamos assumir que a reavaliação completa é aceitável:
    
    print("A reavaliar o melhor indivíduo para obter dados da curva (FPR/TPR)...")
    _, _, history = config['individual_evaluation_func'](best_genotype, config)
    
    
    # 3. Extrair os dados brutos e Plotar
    fpr_final = history.get('fpr') 
    tpr_final = history.get('tpr')
    
    if fpr_final is not None and tpr_final is not None:
        file_name_prefix = f"{config['dataset']}_{config['experiment_name']}_{config['seed']}_FINAL_ROC"
        output_folder = os.path.join(config['output_csv_folder'], 'final_plots')
        
        # Chama o script de plotting (plot_roc.py)
        plot_roc.plot_roc_curve_and_save(fpr_final, tpr_final, best_auc, file_name_prefix, output_folder)
        print(f"Plotagem ROC Final Concluída.")
    else:
        print("Atenção: Não foi possível obter os dados FPR/TPR. Verifique o retorno do sl_evaluation.py.")


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
