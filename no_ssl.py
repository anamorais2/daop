import numpy as np
import torch
import sys
import os
    

def train(model, trainloader, config):
    model.model = model.model.to(config['device'])
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
        

        final_loss = running_loss / len(trainloader)
        final_acc = running_acc / len(trainloader)

        print(f"Epoch {epoch + 1}, Total {i + 1} batches, Loss: {final_loss :.4f}, Accuracy: {final_acc :.4f}")

        hist_loss.append(final_loss)
        hist_acc.append(final_acc)

    print("Finished training")

    return hist_loss, hist_acc

def test(model, testloader, device, confusion_matrix_config):
    model.model = model.model.to(device)
    model.model.eval()
    correct = 0
    total = 0

    if confusion_matrix_config:
        all_labels = []
        all_predicted = []

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model.model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if confusion_matrix_config:
                all_labels.extend(labels.tolist())
                all_predicted.extend(predicted.tolist())

    print(f"Test Accuracy: {correct / total * 100:.2f}%")

    if confusion_matrix_config:
        from torcheval.metrics.functional import multiclass_confusion_matrix
        all_labels = torch.tensor(all_labels)
        all_predicted = torch.tensor(all_predicted)
        conf_matrix = multiclass_confusion_matrix(all_labels, all_predicted, num_classes=confusion_matrix_config['num_classes_downstream'])
        if confusion_matrix_config['print_confusion_matrix']:
            print(f"Confusion Matrix for Test:\n{conf_matrix.tolist()}")

        if confusion_matrix_config['confusion_matrix_folder']:
            os.makedirs(confusion_matrix_config['confusion_matrix_folder'], exist_ok=True)
            confusion_matrix_path = os.path.join(confusion_matrix_config['confusion_matrix_folder'], confusion_matrix_config['confusion_matrix_downstream_file'])
            with open(confusion_matrix_path, 'a') as f:
                f.write(f"{conf_matrix.tolist()}\n")

    return correct / total
    

def evaluate_no_ssl(trainloader, testloader, config):
    model = config['model']()
    hist_loss, hist_acc = train(model, trainloader, config)
    acc = test(model, testloader, config['device'], config['confusion_matrix_config'])
    return acc, acc, {"loss": hist_loss, "acc": hist_acc}
