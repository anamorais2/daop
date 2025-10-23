import os
import sys
import ast
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay


FILE_TYPE = ".txt"

FOLDER1 = "output_confusion_matrix"
NAME1 = "confusion_matrix_pretext_my"

FOLDER2 = "output_confusion_matrix"
NAME2 = "confusion_matrix_pretext_rn"

FONT_SIZE = 24

if FILE_TYPE == ".npy":
    cm1 = np.load(os.path.join(FOLDER1, NAME1))
    cm2 = np.load(os.path.join(FOLDER2, NAME2))
elif FILE_TYPE == ".txt":
    with open(os.path.join(FOLDER1, f"{NAME1}{FILE_TYPE}"), "r") as f:
        cm1_raw = ast.literal_eval(f.readline())

    with open(os.path.join(FOLDER2, f"{NAME2}{FILE_TYPE}"), "r") as f:
        cm2_raw = ast.literal_eval(f.readline())

    cm1 = np.array(cm1_raw)
    cm2 = np.array(cm2_raw)

print(cm1.shape)
print(cm2.shape)

cm_diff = cm1 - cm2

print(cm_diff)

if cm_diff.shape[0] == 4:

    rotations_degrees = ["0°", "90°", "180°", "270°"]

    disp_pretext = ConfusionMatrixDisplay(cm_diff, display_labels=rotations_degrees)
    disp_pretext.plot(text_kw={"fontsize": FONT_SIZE})

    plt.xlabel("True Label", fontsize=FONT_SIZE)
    plt.ylabel("Predicted Label", fontsize=FONT_SIZE)
elif cm_diff.shape[0] == 10:
    cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    disp_downstream = ConfusionMatrixDisplay(cm_diff, display_labels=cifar10_classes)
    disp_downstream.plot(xticks_rotation=60, text_kw={"fontsize": FONT_SIZE})

    plt.xlabel("True Label", fontsize=FONT_SIZE)
    plt.ylabel("Predicted Label", fontsize=FONT_SIZE)
    
else:
    print("Wrong Confusion Matrix Shape")
    sys.exit(1)

plt.subplots_adjust(bottom=0.25, left=0, top=0.95)
plt.xticks(fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.show()