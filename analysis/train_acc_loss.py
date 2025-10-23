import pandas as pd
import numpy as np
import os
import ast
import matplotlib.pyplot as plt

BASE_FOLDER = 'output_csv_optimize_downstream'
EXPERIMENT = 'cifar10_optimize_downstream_0'

# BASE_FOLDER = 'output_csv_optimize_pretext'
# EXPERIMENT = 'cifar10_optimize_pretext_0'

PLOT_VALIDATION = True

individual_index = 0
generation_index = -1

df = pd.read_csv(os.path.join(BASE_FOLDER, f'{EXPERIMENT}.csv'), sep=';')
population = ast.literal_eval(df['population'].iloc[generation_index])
individual = population[individual_index]

pretext_acc_hist = individual[4]['pretext_acc']
pretext_loss_hist = individual[4]['pretext_loss']
pretext_test_acc = individual[2]

downstream_acc_hist = individual[4]['downstream_acc']
downstream_loss_hist = individual[4]['downstream_loss']
downstream_test_acc = individual[1]

# plot 4 graphs


# pretext acc
if pretext_test_acc != -1:
    plt.plot(pretext_acc_hist, label='pretext train acc')
    plt.plot([pretext_test_acc]*len(pretext_acc_hist), label='pretext test acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Evolution of Pretext Accuracy\n' + EXPERIMENT.replace('_', '-'))
    plt.legend()
    plt.show()

    # pretext loss
    plt.plot(pretext_loss_hist, label='pretext train loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Evolution of Pretext Loss\n' + EXPERIMENT.replace('_', '-'))
    plt.legend()
    plt.show()

# downstream acc
plt.plot(downstream_acc_hist, label='downstream train acc')
plt.plot([downstream_test_acc]*len(downstream_acc_hist), label='downstream test acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Evolution of Downstream Accuracy\n' + EXPERIMENT.replace('_', '-'))
plt.legend()
plt.show()

# downstream loss
plt.plot(downstream_loss_hist, label='downstream train loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Evolution of Downstream Loss\n' + EXPERIMENT.replace('_', '-'))
plt.legend()
plt.show()
