import os
import sys
import ast
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay

BASE_FOLDER = "output_confusion_matrix"
BASE_NAME = "confusion_matrix_no_ssl"

FILE_TYPE = ".txt"

SAVE_FILE = "output_" + BASE_NAME

confusion_matrices = []

if not os.path.exists(BASE_FOLDER):
    print(f"Folder {BASE_FOLDER} does not exist")
    sys.exit(1)

with open(os.path.join(BASE_FOLDER, f"{BASE_NAME}{FILE_TYPE}"), "r") as f:
    for line in f:
        confusion_matrices.append(ast.literal_eval(line))

confusion_matrices= np.array(confusion_matrices)

mean_cm = np.mean(confusion_matrices, axis=0)

print(mean_cm)

if SAVE_FILE:
    np.save(os.path.join(BASE_FOLDER, f"{SAVE_FILE}.npy"), mean_cm)

cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

disp_cm = ConfusionMatrixDisplay(mean_cm, display_labels=cifar10_classes)
disp_cm.plot(values_format='.2f')

plt.xlabel("True Label")
plt.ylabel("Predicted Label")

plt.show()


