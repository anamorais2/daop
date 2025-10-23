import numpy as np
import torch
import sys
import os


def rotnet_rotate_images(data):
    rotated_images = []
    rotated_labels = []
    for d in data:
        if type(d) == tuple and len(d) == 2:
            img = d[0]
        else:
            img = d
        for rotation in range(4):
            rotated_img = torch.rot90(img, k=rotation, dims=(1, 2))
            rotated_images.append(rotated_img)
            rotated_labels.append(rotation)
    
    return torch.stack(rotated_images), torch.tensor(rotated_labels)


def rotnet_collate_fn(data):
    rotated_images = []
    rotated_labels = []
    for img, _ in data:
        # img está no formato PyTorch (C, H, W), onde C=1.
        
        # 1. FORÇAR REPLICAÇÃO DE CANAL PARA 3 (para o ResNet)
        # Se a imagem tiver 1 canal, replicamos o canal zero 3 vezes.
        if img.shape[0] == 1:
            img_3ch = img.repeat(3, 1, 1)  # Novo tensor (3, H, W)
        else:
            img_3ch = img
            
        for rotation in range(4):
            # Rotacionar nas dimensões 1 (H) e 2 (W)
            rotated_img = torch.rot90(img_3ch, k=rotation, dims=(1, 2))
            
            rotated_images.append(rotated_img)
            rotated_labels.append(rotation)
    
    return torch.stack(rotated_images), torch.tensor(rotated_labels)


def rotnet_collate_fn_cuda(batch):
    # Rotate 0º
    data = torch.stack([item[0] for item in batch]).to('cuda')
    targets = torch.zeros(data.size(dim=0), dtype=torch.int64, device='cuda')
    # Rotate 90º
    data_90 = torch.rot90(data, dims=(2, 3))
    targets_90 = targets + 1
    # rotate 180º
    data_180 = torch.rot90(data_90, dims=(2, 3))
    targets_180 = targets + 2
    # rotate 270º
    data_270 = torch.rot90(data_180, dims=(2, 3))
    targets_270 = targets + 3

    return (torch.concatenate((data, data_90, data_180, data_270)),
            torch.concatenate((targets, targets_90, targets_180, targets_270)))

def train_pretext(model, trainloader_pretext, config):
    model.model = model.model.to(config['device'])
    model.model.train()
    criterion = model.criterion()
    optimizer = model.optimizer(model.model.parameters())

    hist_loss = []
    hist_acc = []

    for epoch in range(config['pretext_epochs']()):
        running_loss = 0
        running_acc = 0
        for i, data in enumerate(trainloader_pretext, 0):
            if config['track_train_bottleneck']:
                print("Pretext START Epoch", epoch + 1, "Batch", i + 1)

            images, labels = data[0].to(config['device']), data[1].to(config['device'])

            optimizer.zero_grad()
            outputs = model.model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total = labels.size(0)
            accuracy = (predicted == labels).sum() / total

            running_loss += loss.item()
            running_acc += accuracy.item()

            if config['track_train_bottleneck']:
                print("Pretext END Epoch", epoch + 1, "Batch", i + 1)
        

        final_loss = running_loss / len(trainloader_pretext)
        final_acc = running_acc / len(trainloader_pretext)

        print(f"Epoch {epoch + 1}, Total {i + 1} batches, Loss: {final_loss :.4f}, Accuracy: {final_acc :.4f}")

        hist_loss.append(final_loss)
        hist_acc.append(final_acc)

    print("Finished pretext training")

    if config['save_pretext_model']:
        print("Saving pretext model")
        if not os.path.exists(config['save_models_folder']):
            os.makedirs(config['save_models_folder'])
        torch.save(model.model.state_dict(), os.path.join(config['save_models_folder'], f"{config['save_pretext_model']}_{config['seed']}.pt"))

    return hist_loss, hist_acc

def test_pretext(model, testloader_pretext, device, confusion_matrix_config):
    model.model = model.model.to(device)
    model.model.eval()
    correct = 0
    total = 0

    if confusion_matrix_config:
        all_labels = []
        all_predicted = []

    with torch.no_grad():
        for data in testloader_pretext:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model.model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if confusion_matrix_config:
                all_labels.extend(labels.tolist())
                all_predicted.extend(predicted.tolist())

    print(f"Pretext Test Accuracy: {correct / total * 100:.2f}%")

    if confusion_matrix_config:
        from torcheval.metrics.functional import multiclass_confusion_matrix
        all_labels = torch.tensor(all_labels)
        all_predicted = torch.tensor(all_predicted)
        conf_matrix = multiclass_confusion_matrix(all_labels, all_predicted, num_classes=confusion_matrix_config['num_classes_pretext'])
        if confusion_matrix_config['print_confusion_matrix']:
            print(f"Confusion Matrix for Pretext Test:\n{conf_matrix.tolist()}")

        if confusion_matrix_config['confusion_matrix_folder']:
            os.makedirs(confusion_matrix_config['confusion_matrix_folder'], exist_ok=True)
            confusion_matrix_path = os.path.join(confusion_matrix_config['confusion_matrix_folder'], confusion_matrix_config['confusion_matrix_pretext_file'])
            with open(confusion_matrix_path, 'a') as f:
                f.write(f"{conf_matrix.tolist()}\n")

    return correct / total

def train_downstream(model, trainloader_downstream, config):
    if model.mode != "downstream":
        model.switch_to_downstream()

    model.model.to(config['device'])
    model.model.train()
    criterion = model.criterion()
    optimizer = model.optimizer(model.model.fc.parameters())

    hist_loss = []
    hist_acc = []

    for epoch in range(config['downstream_epochs']()):
        running_loss = 0
        running_acc = 0
        for i, data in enumerate(trainloader_downstream, 0):
            if config['track_train_bottleneck']:
                print("Downstream START Epoch", epoch + 1, "Batch", i + 1)

            images, labels = data[0].to(config['device']), data[1].to(config['device'])
            images = images.squeeze(1)

            optimizer.zero_grad()
            outputs = model.model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            total = labels.size(0)
            accuracy = (predicted == labels).sum() / total

            running_loss += loss.item()
            running_acc += accuracy.item()

            if config['track_train_bottleneck']:
                print("Downstream END Epoch", epoch + 1, "Batch", i + 1)

        final_loss = running_loss / len(trainloader_downstream)
        final_acc = running_acc / len(trainloader_downstream)

        print(f"Epoch {epoch + 1}, Total {i + 1} batches, Loss: {final_loss :.4f}, Accuracy: {final_acc :.4f}")

        hist_loss.append(final_loss)
        hist_acc.append(final_acc)

    print("Finished downstream training")

    if config['save_downstream_model']:
        print("Saving downstream model")
        if not os.path.exists(config['save_models_folder']):
            os.makedirs(config['save_models_folder'])
        torch.save(model.model.state_dict(), os.path.join(config['save_models_folder'], f"{config['save_downstream_model']}_{config['seed']}.pt"))

    return hist_loss, hist_acc

def test_downstream(model, testloader_downstream, device, confusion_matrix_config):
    model.model.to(device)
    model.model.eval()
    correct = 0
    total = 0

    if confusion_matrix_config:
        all_labels = []
        all_predicted = []

    with torch.no_grad():
        for data in testloader_downstream:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model.model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if confusion_matrix_config:
                all_labels.extend(labels.tolist())
                all_predicted.extend(predicted.tolist())

    print(f"Downstream Test Accuracy: {correct / total * 100:.2f}%")

    if confusion_matrix_config:
        from torcheval.metrics.functional import multiclass_confusion_matrix
        all_labels = torch.tensor(all_labels)
        all_predicted = torch.tensor(all_predicted)
        conf_matrix = multiclass_confusion_matrix(all_labels, all_predicted, num_classes=confusion_matrix_config['num_classes_downstream'])
        if confusion_matrix_config['print_confusion_matrix']:
            print(f"Confusion Matrix for Downstream Test:\n{conf_matrix.tolist()}")

        if confusion_matrix_config['confusion_matrix_folder']:
            os.makedirs(confusion_matrix_config['confusion_matrix_folder'], exist_ok=True)
            confusion_matrix_path = os.path.join(confusion_matrix_config['confusion_matrix_folder'], confusion_matrix_config['confusion_matrix_downstream_file'])
            with open(confusion_matrix_path, 'a') as f:
                f.write(f"{conf_matrix.tolist()}\n")

    return correct / total
    

def evaluate_rotnet(trainloader_pretext, trainloader_downstream, testloader_pretext, testloader_downstream, config):
    model = config['model']()

    if config['pretrained_pretext_model']:
        print("Using pretrained pretext model")
        model.load_weights_from_path(config['pretrained_pretext_model'])

        pretext_hist_loss, pretext_hist_acc = (["Used pretrained pretext model"], ["Used pretrained pretext model"])
        pretext_acc = -1
    else:
        pretext_hist_loss, pretext_hist_acc = train_pretext(model, trainloader_pretext, config)
        pretext_acc = test_pretext(model, testloader_pretext, config['device'], config['confusion_matrix_config'])

    model.switch_to_downstream()
    downstream_hist_loss, downstream_hist_acc = train_downstream(model, trainloader_downstream, config)
    downstream_acc = test_downstream(model, testloader_downstream, config['device'], config['confusion_matrix_config'])
    return downstream_acc, pretext_acc, {"pretext_loss": pretext_hist_loss, "pretext_acc": pretext_hist_acc, 
                                         "downstream_loss": downstream_hist_loss, "downstream_acc": downstream_hist_acc}
