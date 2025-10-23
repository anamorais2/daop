import os
import sys

from sklearn.metrics import ConfusionMatrixDisplay
import torch
import numpy as np
import matplotlib.pyplot as plt
from net_models_torch import TrainResNet18Grad
import configs.config_visualize_da as config_visualize_da

config = config_visualize_da.config

FONT_SIZE = 24
MODE = "downstream"

if MODE == "downstream":
    CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    model_paths = ["my_downstream_11.pt", "rn_100_downstream_5.pt", "na_100_downstream_0.pt"]
    display_names = ["Evolved Augments", "RotNet Augments", "No Augments"]

    cms = []

    for display_name, model_path in zip(display_names, model_paths):
        model_instance = TrainResNet18Grad()
        model_instance.switch_to_downstream()
        model_instance.load_weights_from_path(os.path.join("models", model_path))

        model = model_instance.model.to(config['device'])
        model.eval()

        norm_dataset_vars = config['load_dataset_func']([[],[]], config)
        test_loader = config['data_loader_func'](*norm_dataset_vars, config)[3]
        
        correct = 0
        total = 0
        all_labels = []
        all_predicted = []
        with torch.no_grad():
            for data in test_loader:
                images, labels = data[0].to(config['device']), data[1].to(config['device'])
                print(images.shape, images.dtype)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_labels.extend(labels.tolist())
                all_predicted.extend(predicted.tolist())

        print(f"Downstream Test Accuracy: {correct / total * 100:.2f}%")

        from torcheval.metrics.functional import multiclass_confusion_matrix
        all_labels = torch.tensor(all_labels)
        all_predicted = torch.tensor(all_predicted)
        cm = multiclass_confusion_matrix(all_labels, all_predicted, num_classes=config['num_classes_downstream']).numpy()
        print(f"Confusion Matrix for Downstream Test:\n{cm.tolist()}")

        disp_downstream = ConfusionMatrixDisplay(cm, display_labels=CIFAR10_CLASSES)
        disp_downstream.plot(xticks_rotation=60, text_kw={"fontsize": FONT_SIZE})
        plt.subplots_adjust(bottom=0.25, left=0, top=0.95)
        plt.xticks(fontsize=FONT_SIZE)
        plt.yticks(fontsize=FONT_SIZE)
        plt.xlabel("True Label", fontsize=FONT_SIZE)
        plt.ylabel("Predicted Label", fontsize=FONT_SIZE)
        plt.title(f"Confusion Matrix for {display_name}", fontsize=FONT_SIZE)
        plt.show()

        cms.append(cm)

    if len(model_paths) > 1:
        for i in range(1, len(cms)):
            cm_diff = cms[0] - cms[i]

            print(cm_diff)

            disp_downstream = ConfusionMatrixDisplay(cm_diff, display_labels=CIFAR10_CLASSES)
            disp_downstream.plot(xticks_rotation=60, text_kw={"fontsize": FONT_SIZE})

            plt.xlabel("True Label", fontsize=FONT_SIZE)
            plt.ylabel("Predicted Label", fontsize=FONT_SIZE)

            plt.subplots_adjust(bottom=0.25, left=0, top=0.95)
            plt.xticks(fontsize=FONT_SIZE)
            plt.yticks(fontsize=FONT_SIZE)
            plt.title(f"Confusion Matrix Difference between {display_names[0]} and {display_names[i]}", fontsize=FONT_SIZE)
            plt.show()
elif MODE=="pretext":
    ROTNET_CLASSES = ["0", "90", "180", "270"]
    model_paths = ["my_pretext_11.pt", "rn_100_pretext_5.pt", "na_100_pretext_0.pt"]
    display_names = ["Evolved Augments", "RotNet Augments", "No Augments"]

    cms = []

    for display_name, model_path in zip(display_names, model_paths):
        model_instance = TrainResNet18Grad()
        model_instance.load_weights_from_path(os.path.join("models", model_path))

        model = model_instance.model.to(config['device'])
        model.eval()

        norm_dataset_vars = config['load_dataset_func']([[],[]], config)
        test_loader = config['data_loader_func'](*norm_dataset_vars, config)[2]
        
        correct = 0
        total = 0
        all_labels = []
        all_predicted = []
        with torch.no_grad():
            for data in test_loader:
                images, labels = data[0].to(config['device']), data[1].to(config['device'])
                print(images.shape, images.dtype)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_labels.extend(labels.tolist())
                all_predicted.extend(predicted.tolist())

        print(f"Downstream Test Accuracy: {correct / total * 100:.2f}%")

        from torcheval.metrics.functional import multiclass_confusion_matrix
        all_labels = torch.tensor(all_labels)
        all_predicted = torch.tensor(all_predicted)
        cm = multiclass_confusion_matrix(all_labels, all_predicted, num_classes=config['num_classes_pretext']).numpy()
        print(f"Confusion Matrix for Downstream Test:\n{cm.tolist()}")

        disp_downstream = ConfusionMatrixDisplay(cm, display_labels=ROTNET_CLASSES)
        disp_downstream.plot(text_kw={"fontsize": FONT_SIZE})
        plt.subplots_adjust(bottom=0.25, left=0, top=0.95)
        plt.xticks(fontsize=FONT_SIZE)
        plt.yticks(fontsize=FONT_SIZE)
        plt.xlabel("True Label", fontsize=FONT_SIZE)
        plt.ylabel("Predicted Label", fontsize=FONT_SIZE)
        plt.title(f"Confusion Matrix for {display_name}", fontsize=FONT_SIZE)
        plt.show()

        cms.append(cm)

    if len(model_paths) > 1:
        for i in range(1, len(cms)):
            cm_diff = cms[0] - cms[i]

            print(cm_diff)

            disp_downstream = ConfusionMatrixDisplay(cm_diff, display_labels=ROTNET_CLASSES)
            disp_downstream.plot(xticks_rotation=60, text_kw={"fontsize": FONT_SIZE})

            plt.xlabel("True Label", fontsize=FONT_SIZE)
            plt.ylabel("Predicted Label", fontsize=FONT_SIZE)

            plt.subplots_adjust(bottom=0.25, left=0, top=0.95)
            plt.xticks(fontsize=FONT_SIZE)
            plt.yticks(fontsize=FONT_SIZE)
            plt.title(f"Confusion Matrix Difference between {display_names[0]} and {display_names[i]}", fontsize=FONT_SIZE)
            plt.show()