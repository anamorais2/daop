import numpy as np
import torch
import sys
import os
from sklearn.metrics import roc_curve, auc
from analysis import plot_roc
from analysis.plot_roc import plot_roc_curve_and_save
from torcheval.metrics.functional import multiclass_confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_curve, auc, roc_auc_score


def train_sl(model, trainloader, config):
    
    model.model.to(config['device'])
    model.model.train()
    criterion = model.criterion()
    optimizer = model.optimizer(model.model.parameters()) 

    hist_loss = []
    hist_acc = []
        
    for epoch in range(config['epochs']):
        running_loss = 0
        running_acc = 0
        
        for i, data in enumerate(trainloader, 0):
            images, labels = data[0].to(config['device']), data[1].to(config['device'])

            # Ensure 3 Channels (Channel replication, essential for ResNet)
            if images.shape[1] == 1:
                images = images.repeat(1, 3, 1, 1) 

            optimizer.zero_grad()
            outputs = model.model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            total = labels.size(0)
            accuracy = (predicted == labels).sum().item() / total 

            running_loss += loss.item()
            running_acc += accuracy
        
        final_loss = running_loss / len(trainloader)
        final_acc = running_acc / len(trainloader)

        print(f"Epoch {epoch + 1}, Total {i + 1} batches, Loss: {final_loss :.4f}, Accuracy: {final_acc :.4f}")

        hist_loss.append(final_loss)
        hist_acc.append(final_acc)

    print("Finished Supervised Learning (SL) training")
    return hist_loss, hist_acc

def run_inference(model, loader, device, confusion_matrix_config=None, config=None):
    
    model.model.eval()
    correct = 0
    total = 0

    all_labels_list = []
    all_outputs_list = [] 
    
    # Listas para Matriz de Confusão (apenas se config for passada)
    all_predicted_cm = []
    all_labels_cm = []

    # Obter número de classes se config estiver disponível
    num_classes = 2
    if config:
        if confusion_matrix_config:
             num_classes = confusion_matrix_config.get('num_classes_downstream', config.get('num_classes', 2))
        else:
             num_classes = config.get('num_classes_downstream', config.get('num_classes', 2))

    with torch.no_grad():
        for data in loader:
            images, labels = data[0].to(device), data[1].to(device)
            
            if images.shape[1] == 1:
                images = images.repeat(1, 3, 1, 1)

            outputs = model.model(images)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels_list.append(labels.cpu())
            all_outputs_list.append(outputs.cpu())

            if confusion_matrix_config:
                all_predicted_cm.extend(predicted.cpu().tolist())
                all_labels_cm.extend(labels.cpu().tolist())

    acc = correct / total
    
    all_labels_tensor = torch.cat(all_labels_list)
    all_outputs_tensor = torch.cat(all_outputs_list)
    all_probs_tensor = torch.softmax(all_outputs_tensor, dim=1)

    fpr, tpr = None, None
    roc_auc = 0.0
    auc_std = 0.0

    try:
        if num_classes == 2:
        
            probs_binary = all_probs_tensor[:, 1].numpy()
            labels_binary = all_labels_tensor.numpy()
            
            if len(np.unique(labels_binary)) > 1:
                roc_auc = roc_auc_score(labels_binary, probs_binary)
                fpr, tpr, _ = roc_curve(labels_binary, probs_binary)
            else:
                roc_auc = 0.5 
        else:
            roc_auc_per_class = roc_auc_score(
                all_labels_tensor.numpy(),
                all_probs_tensor.numpy(),
                multi_class='ovr',
                average=None 
            )
            roc_auc = np.mean(roc_auc_per_class)
            auc_std = np.std(roc_auc_per_class)

    except ValueError:
        roc_auc = -1.0 

    if confusion_matrix_config:
        all_labels_cm_tensor = torch.tensor(all_labels_cm)
        all_predicted_cm_tensor = torch.tensor(all_predicted_cm)
        
        conf_matrix = multiclass_confusion_matrix(all_labels_cm_tensor, all_predicted_cm_tensor, num_classes=num_classes)
        
        if confusion_matrix_config.get('print_confusion_matrix', False):
            print(f"Confusion Matrix:\n{conf_matrix.tolist()}")

        if confusion_matrix_config.get('confusion_matrix_folder'):
            folder = confusion_matrix_config['confusion_matrix_folder']
            os.makedirs(folder, exist_ok=True)
            file_name = f"CM_{config['dataset']}_{config['experiment_name']}_{config['seed']}.txt"
            confusion_matrix_path = os.path.join(folder, file_name)
            with open(confusion_matrix_path, 'a') as f:
                gen = config.get('generation', 'Final')
                f.write(f"Generation: {gen}\n{conf_matrix.tolist()}\n\n")

    return acc, roc_auc, auc_std, fpr, tpr



def test_sl_multi(model, testloader, device, confusion_matrix_config, config):
    
    model.model.to(device)
    model.model.eval()
    correct = 0
    total = 0

    all_labels_list = []
    all_outputs_list = [] 
    
    if confusion_matrix_config:
        all_predicted_cm = [] 
        all_labels_cm = []

    # Get the number of classes (2 for BreastMNIST, 14 for ChestMNIST)
    if confusion_matrix_config:
         num_classes = confusion_matrix_config.get('num_classes_downstream', config.get('num_classes', 2))
    else:
         num_classes = config.get('num_classes_downstream', config.get('num_classes', 2))

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            
            if images.shape[1] == 1:
                images = images.repeat(1, 3, 1, 1)

            outputs = model.model(images)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if confusion_matrix_config:
                all_predicted_cm.extend(predicted.cpu().tolist())
                all_labels_cm.extend(labels.cpu().tolist())
            
            all_labels_list.append(labels.cpu())
            all_outputs_list.append(outputs.cpu())


    sl_acc = correct / total
    print(f"SL Test Accuracy: {sl_acc * 100:.2f}%")

    all_labels_tensor = torch.cat(all_labels_list)
    all_outputs_tensor = torch.cat(all_outputs_list)
    all_probs_tensor = torch.softmax(all_outputs_tensor, dim=1)

    fpr, tpr = None, None

    if num_classes == 2:
        # 1. BINARY (ex: BreastMNIST)
        # Extract probabilities of the positive class (index 1)
        probs_binary = all_probs_tensor[:, 1].numpy()
        labels_binary = all_labels_tensor.numpy()
        
        roc_auc = roc_auc_score(labels_binary, probs_binary)
        auc_std = 0.0  # Placeholder, can compute std if needed
       
        fpr, tpr, thresholds = roc_curve(labels_binary, probs_binary)
        
        print(f"SL Test AUC (2-Class): {roc_auc:.4f}")
        
    else:
        # 2. MULTI-CLASS (e.g. ChestMNIST)
        # Use roc_auc_score with One-vs-Rest (ovr) and 'weighted' average
        try:
            roc_auc_per_class = roc_auc_score(
                all_labels_tensor.numpy(),
                all_probs_tensor.numpy(),
                multi_class='ovr',
                average=None 
            )

            roc_auc = np.mean(roc_auc_per_class)
            auc_std = np.std(roc_auc_per_class)

            print(f"SL Test AUC (Multi-Class, OVR Weighted): {roc_auc:.4f} ± {auc_std:.4f}")
        except ValueError as e:
            print(f"WARNING: Could not compute Multi-Class AUC. {e}")
            roc_auc = -1.0 # Error value
            auc_std = 0.0

            # FPR/TPR curve data is not directly applicable in multi-class
        fpr, tpr = None, None

    if confusion_matrix_config:
        all_labels_cm_tensor = torch.tensor(all_labels_cm)
        all_predicted_cm_tensor = torch.tensor(all_predicted_cm)
        
        conf_matrix = multiclass_confusion_matrix(all_labels_cm_tensor, all_predicted_cm_tensor, num_classes=num_classes)
        
        if confusion_matrix_config.get('print_confusion_matrix', False):
            print(f"Confusion Matrix (SL):\n{conf_matrix.tolist()}")

        if confusion_matrix_config.get('confusion_matrix_folder'):
            folder = confusion_matrix_config['confusion_matrix_folder']
            os.makedirs(folder, exist_ok=True)
            file_name = f"CM_{config['dataset']}_{config['experiment_name']}_{config['seed']}.txt"
            confusion_matrix_path = os.path.join(folder, file_name)
            with open(confusion_matrix_path, 'a') as f:
                f.write(f"Generation: {config.get('generation', 'Final')}\n{conf_matrix.tolist()}\n\n")

    return sl_acc, roc_auc, auc_std, fpr, tpr


def evaluate_sl(trainloader, valloader, testloader, config):

    num_classes = config.get('num_classes_downstream', config.get('num_classes', 2))
    model = config['model'](num_classes_downstream=num_classes)

    sl_hist_loss, sl_hist_acc = train_sl(model, trainloader, config)

    val_acc, val_auc, val_auc_std, fpr_val, tpr_val = run_inference(model, valloader, config['device'], config['confusion_matrix_config'], config)

    test_acc, test_auc, test_auc_std, fpr_test, tpr_test = run_inference(model, testloader, config['device'], config['confusion_matrix_config'], config)


    print(f"  -> Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f} | Test AUC: {test_auc:.4f}")

    history = {
        "sl_hist_loss": sl_hist_loss,
        "sl_hist_acc": sl_hist_acc,
        "val_acc": val_acc,
        "val_auc": val_auc,
        "test_acc": test_acc,
        "test_auc": test_auc,
        "test_auc_std": test_auc_std,
        "fpr": fpr_test,
        "tpr": tpr_test
    }

    # The EA expects 3 returns. pretext_acc is now unused (-1).
    # Return: final SL accuracy, a placeholder pretext accuracy (-1), and the history dict.
    return val_acc, -1, history