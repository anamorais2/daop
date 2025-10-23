import os
import sys
import ast
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay

BASE_FOLDER = "output_confusion_matrix"
BASE_NAME = "confusion_matrix_na"
PRETEXT_SUFFIX = "pretext"
DOWNSTREAM_SUFFIX = "downstream"

FILE_TYPE = ".txt"

SAVE_FILE = "output_" + BASE_NAME

confusion_matrices_pretext = []
confusion_matrices_downstream = []

if not os.path.exists(BASE_FOLDER):
    print(f"Folder {BASE_FOLDER} does not exist")
    sys.exit(1)

with open(os.path.join(BASE_FOLDER, f"{BASE_NAME}_{PRETEXT_SUFFIX}{FILE_TYPE}"), "r") as f:
    for line in f:
        confusion_matrices_pretext.append(ast.literal_eval(line))

with open(os.path.join(BASE_FOLDER, f"{BASE_NAME}_{DOWNSTREAM_SUFFIX}{FILE_TYPE}"), "r") as f:
    for line in f:
        confusion_matrices_downstream.append(ast.literal_eval(line))

confusion_matrices_pretext = np.array(confusion_matrices_pretext)
confusion_matrices_downstream = np.array(confusion_matrices_downstream)

mean_pretext = np.mean(confusion_matrices_pretext, axis=0)
mean_downstream = np.mean(confusion_matrices_downstream, axis=0)

print(mean_pretext)
print(mean_downstream)

if SAVE_FILE:
    np.save(os.path.join(BASE_FOLDER, f"{SAVE_FILE}_{PRETEXT_SUFFIX}.npy"), mean_pretext)
    np.save(os.path.join(BASE_FOLDER, f"{SAVE_FILE}_{DOWNSTREAM_SUFFIX}.npy"), mean_downstream)

rotations_degrees = ["0°", "90°", "180°", "270°"]

disp_pretext = ConfusionMatrixDisplay(mean_pretext, display_labels=rotations_degrees)
disp_pretext.plot(values_format='.2f')

plt.xlabel("True Label")
plt.ylabel("Predicted Label")

plt.show()

cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

disp_downstream = ConfusionMatrixDisplay(mean_downstream, display_labels=cifar10_classes)
disp_downstream.plot(values_format='.2f')

plt.xlabel("True Label")
plt.ylabel("Predicted Label")

plt.show()


