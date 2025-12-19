import os
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt

FONT_SIZE = 14
plt.rcParams.update({'font.size': FONT_SIZE})

def plot_evolution_metric(folder_path, metric_col="test_acc", metric_label="Test Accuracy", experiment_name="Experiment", baseline=None):
  
    
    # 1. Encontrar ficheiros CSV
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    if not all_files:
        print(f"ERRO: Nenhuns ficheiros CSV encontrados em: {folder_path}")
        return

    print(f"Processando {len(all_files)} seeds para a métrica: {metric_col}")

    data_frames = []

    for filename in all_files:
        try:
            # Lê apenas a geração e a métrica desejada
            df = pd.read_csv(
                filename, 
                sep=';', 
                usecols=['generation', metric_col], 
                on_bad_lines='skip'
            )
            
            # Converter para numérico e forçar erros a NaN
            df['generation'] = pd.to_numeric(df['generation'], errors='coerce')
            df[metric_col] = pd.to_numeric(df[metric_col], errors='coerce')
            
            # Limpar dados inválidos (NaNs)
            df = df.dropna()
            
            # === CORREÇÃO DO ERRO DE DUPLICADOS ===
            # Se houver duas linhas para a mesma geração, mantém a última (keep='last')
            # Isto resolve o "cannot reindex on an axis with duplicate labels"
            if df.duplicated(subset=['generation']).any():
                print(f"Aviso: Duplicados encontrados e removidos em {os.path.basename(filename)}")
                df = df.drop_duplicates(subset=['generation'], keep='last')
            # ======================================
            
            if not df.empty:
                df = df.sort_values('generation')
                df = df.set_index('generation')
                
                # Guarda apenas a série da métrica
                data_frames.append(df[metric_col])
            else:
                print(f"Aviso: Ficheiro {os.path.basename(filename)} vazio após limpeza.")
                
        except ValueError as ve:
            print(f"Erro de coluna em {os.path.basename(filename)}: A coluna '{metric_col}' não existe?")
        except Exception as e:
            print(f"Erro ao ler {os.path.basename(filename)}: {e}")

    if not data_frames:
        print("Nenhum dado válido carregado. Verifica os nomes das colunas no CSV.")
        return

    # 2. Consolidar Dados (Agora seguro contra duplicados)
    try:
        df_all = pd.concat(data_frames, axis=1)
    except ValueError as e:
        print(f"Erro fatal ao juntar DataFrames: {e}")
        return

    df_all = df_all.ffill() # Preencher lacunas com o valor anterior

    # 3. Calcular Métricas ROB (Run Overall Best)
    df_cummax = df_all.cummax()
    
    rob_mean = df_cummax.mean(axis=1)
    rob_std = df_cummax.std(axis=1)

    # OMF (Overall Max Fitness/Metric)
    omf_curve = df_all.max(axis=1).cummax()

    # 4. Plotting
    plt.figure(figsize=(10, 6))
    
    generations = rob_mean.index
    num_seeds = df_all.shape[1]

    # Plot OMF
    plt.plot(generations, omf_curve, label=f'Max {metric_label} (Best Seed)', 
             color='green', linestyle='--', linewidth=2, alpha=0.9)

    # Plot ROB
    plt.plot(generations, rob_mean, label=f'Average {metric_label} (Robustness)', 
             color='blue', linewidth=2)
    
    # Sombra do Desvio Padrão
    plt.fill_between(generations, rob_mean - rob_std, rob_mean + rob_std, 
                     color='blue', alpha=0.15, label='Standard Deviation')

    # Baseline
    if baseline is not None:
        plt.axhline(y=baseline, color='red', linestyle='-.', linewidth=2, 
                    label=f'Baseline ({baseline})', alpha=0.8)

    plt.title(f'Evolution of {metric_label} - {experiment_name}\n(Average of {num_seeds} seeds)', fontsize=16)
    plt.xlabel('Generation')
    plt.ylabel(metric_label)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    # Guardar
    output_filename = f"evolution_{metric_col}.png"
    output_path = os.path.join(folder_path, output_filename)
    plt.savefig(output_path, dpi=300)
    print(f"Gráfico guardado em: {output_path}")
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        folder = sys.argv[1]
        exp_name = sys.argv[2] if len(sys.argv) > 2 else "DAOP Experiment"
        baseline_val = float(sys.argv[3]) if len(sys.argv) > 3 else None
        
        # --- CHOOSE WHAT TO PLOT HERE ---
        
        # 1. Generate Test Accuracy evolution plot
        print("\n--- Generating TEST ACCURACY plot ---")
        plot_evolution_metric(folder, metric_col="acc_test", metric_label="Accuracy Test", 
                              experiment_name=exp_name, baseline=baseline_val)
        
        # 2. Generate Test AUC evolution plot (useful for thesis)
        print("\n--- Generating TEST AUC plot ---")
        # Note: baseline for AUC is different from Accuracy, pass None or adjust as needed
        plot_evolution_metric(folder, metric_col="auc_test", metric_label="Auc Test", 
                              experiment_name=exp_name, baseline=0.913) 
        
    else:
        print("Usage: python graphs_medmnist.py <path_to_csv_folder> [Experiment_Name] [Baseline_Value]")