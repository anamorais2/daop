import matplotlib.pyplot as plt
import os
import numpy as np

def plot_roc_curve_and_save(fpr, tpr, roc_auc, file_name_prefix, output_folder):
    
    os.makedirs(output_folder, exist_ok=True)
    file_path = os.path.join(output_folder, f"ROC_{file_name_prefix}.png")
    
    plt.figure()
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC (AUC = {roc_auc:.4f})')
             
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Curva ROC - {file_name_prefix}')
    plt.legend(loc="lower right")

    plt.savefig(file_path)
    plt.close()
    
    print(f"ROC curve saved to {file_path}")