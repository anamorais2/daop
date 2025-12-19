import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


labels = [
    'BreastMNIST\n(ResNet-18)', 
    'BreastMNIST\n(ResNet-50)', 
    'DermaMNIST\n(ResNet-18)', 
    'DermaMNIST\n(ResNet-50)',
    'PneumoniaMNIST\n(ResNet-18)'
]

my_auc_scores = [0.929, 0.920, 0.937, 0.928, 0.983] 
my_acc_scores = [0.872, 0.878, 0.777, 0.765, 0.873] 

baseline_auc = [0.901, 0.857, 0.917, 0.913, 0.944]
baseline_acc = [0.863, 0.812, 0.735, 0.735, 0.854]



x = np.arange(len(labels))  
width = 0.35 


fig, ax = plt.subplots(figsize=(14, 6))


rects1 = ax.bar(x - width/2, my_auc_scores, width, label='DAOP AUC', color='#6c8ebf', edgecolor='black')
rects2 = ax.bar(x + width/2, my_acc_scores, width, label='DAOP AAC', color='#d5e8d4', edgecolor='black')


def plot_baselines(x_pos, values, offset):
   
    for i, val in enumerate(values):
    
        center_of_bar = x_pos[i] + offset
        x_min = center_of_bar - width/2
        x_max = center_of_bar + width/2
        
        
        if val > 0:
            ax.hlines(y=val, xmin=x_min, xmax=x_max, 
                      colors='red', linestyles='dashed', linewidth=2, zorder=5)

plot_baselines(x, baseline_auc, -width/2) 
plot_baselines(x, baseline_acc, +width/2) 



custom_line = Line2D([0], [0], color='red', lw=2, linestyle='--', label='Baseline MedMnist')

handles, _ = ax.get_legend_handles_labels()
handles.append(custom_line)

ax.legend(handles=handles, loc='lower right', frameon=True)

# Configuração dos Eixos
ax.set_ylabel('Performance Score', fontsize=12, fontweight='bold')
ax.set_title('Performance Comparison: DAOP vs. Baseline MedMnist', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylim(0, 1.15)  

ax.grid(axis='y', linestyle='--', alpha=0.5)

ax.bar_label(rects1, padding=3, fmt='%.3f', fontsize=9)
ax.bar_label(rects2, padding=3, fmt='%.3f', fontsize=9)

plt.tight_layout()


# plt.savefig('comparison_results.pdf', format='pdf', bbox_inches='tight')

plt.show()