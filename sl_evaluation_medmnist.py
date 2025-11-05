import numpy as np
import torch
import sys
import os
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_curve, auc, roc_auc_score
from torcheval.metrics.functional import multiclass_confusion_matrix



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
            
            # Garante 3 Canais (Replicação de canal, essencial para ResNet)
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



def test_sl(model, testloader, device, confusion_matrix_config, config):
    device = config['device']
    confusion_matrix_config = config.get('confusion_matrix_config') 

    model.model.to(device)
    model.model.eval()
    correct = 0
    total = 0


    all_labels_sl = []
    all_probs_sl = []
    

    if confusion_matrix_config:
        all_labels_cm = []
        all_predicted_cm = []

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            
            # Garantir 3 Canais (Replicação de canal)
            if images.shape[1] == 1:
                images = images.repeat(1, 3, 1, 1)

            outputs = model.model(images)
            
        
            probs = torch.softmax(outputs, dim=1)[:, 1] 
            all_probs_sl.extend(probs.cpu().numpy())
            all_labels_sl.extend(labels.cpu().numpy()) 

        
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

 
            if confusion_matrix_config:
                all_labels_cm.extend(labels.cpu().tolist())
                all_predicted_cm.extend(predicted.cpu().tolist())

    sl_acc = correct / total
    print(f"SL Test Accuracy: {sl_acc * 100:.2f}%")

    fpr, tpr, thresholds = roc_curve(all_labels_sl, all_probs_sl)
    roc_auc = auc(fpr, tpr)
    print(f"SL Test AUC: {roc_auc:.4f}")
 

    if confusion_matrix_config:
        all_labels_cm = torch.tensor(all_labels_cm)
        all_predicted_cm = torch.tensor(all_predicted_cm)
        
        # Obter o número de classes do config (o 2 do BreastMNIST)
        num_classes = confusion_matrix_config.get('num_classes_downstream', config.get('num_classes', 2))
        
        conf_matrix = multiclass_confusion_matrix(all_labels_cm, all_predicted_cm, num_classes=num_classes)
        
        if confusion_matrix_config.get('print_confusion_matrix', False):
            print(f"Confusion Matrix (SL):\n{conf_matrix.tolist()}")

        if confusion_matrix_config.get('confusion_matrix_folder'):
            folder = confusion_matrix_config['confusion_matrix_folder']
            os.makedirs(folder, exist_ok=True)
            
            file_name = f"CM_{config['dataset']}_{config['experiment_name']}_{config['seed']}.txt"
            confusion_matrix_path = os.path.join(folder, file_name)
            
            with open(confusion_matrix_path, 'a') as f:
                f.write(f"Generation: {config.get('generation', 'Final')}\n{conf_matrix.tolist()}\n\n")

    return sl_acc, roc_auc, fpr, tpr

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

    # Obter o número de classes (2 para BreastMNIST, 14 para ChestMNIST)
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
        # 1. BINÁRIO (ex: BreastMNIST)
        # Extrair probabilidades da classe positiva (índice 1)
        probs_binary = all_probs_tensor[:, 1].numpy()
        labels_binary = all_labels_tensor.numpy()
        
        # Calcular AUC (escalar)
        roc_auc = roc_auc_score(labels_binary, probs_binary)
        
        # Calcular dados da Curva (FPR/TPR) para o plot final
        fpr, tpr, thresholds = roc_curve(labels_binary, probs_binary)
        
        print(f"SL Test AUC (Binário): {roc_auc:.4f}")
        
    else:
        # 2. MULTI-CLASSE (ex: ChestMNIST)
        # Usamos roc_auc_score com One-vs-Rest (ovr) e média 'weighted'
        try:
            roc_auc = roc_auc_score(
                all_labels_tensor.numpy(),
                all_probs_tensor.numpy(),
                multi_class='ovr',
                average='weighted' 
            )
            print(f"SL Test AUC (Multi-Classe, OVR Ponderado): {roc_auc:.4f}")
        except ValueError as e:
            print(f"AVISO: Não foi possível calcular o AUC Multi-Classe. {e}")
            roc_auc = -1.0 # Valor de erro
            
            # Dados da curva FPR/TPR não são diretamente aplicáveis em multi-classe
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

    return sl_acc, roc_auc, fpr, tpr


def evaluate_sl(trainloader, testloader, config):
    model = config['model']() 

    sl_hist_loss, sl_hist_acc = train_sl(model, trainloader, config)

    sl_acc, sl_auc, fpr, tpr = test_sl_multi(model, testloader, config['device'], config['confusion_matrix_config'], config)

    # O EA espera 3 retornos. O pretext_acc agora é inútil (-1)
    # Retornamos a accuracy final, uma accuracy de pretext inútil (-1), e o histórico.
    return sl_acc, -1, {"sl_loss": sl_hist_loss, 
                        "sl_acc": sl_hist_acc, 
                        "sl_auc": sl_auc,
                        "fpr": fpr,
                        "tpr": tpr}
