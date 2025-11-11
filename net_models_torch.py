import torch
import torch.nn as nn

import torchvision


class ProjectorBlockResNet18(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ProjectorBlockResNet18, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class TrainResNet18():
    def __init__(self, num_classes_downstream=2):
        self.model = torchvision.models.resnet18()
        self.criterion = nn.CrossEntropyLoss
        self.optimizer = lambda params: torch.optim.Adam(params, lr=0.001)
        self.num_classes_downstream = num_classes_downstream
        self.mode = "pretext"

    def switch_to_downstream(self):
        if self.mode == "downstream":
            return

        self.model.layer3 = nn.Identity()
        self.model.layer4 = nn.Identity()

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc = nn.Sequential(
            ProjectorBlockResNet18(128, 256),
            ProjectorBlockResNet18(256, 256),
            ProjectorBlockResNet18(256, self.num_classes_downstream),
            nn.Softmax(dim=1)
        )

        self.mode = "downstream"

        print("Switched model to downstream mode")


    def load_weights_from_path(self, path):
        self.model.load_state_dict(torch.load(path))

    def __str__(self):
        return "RotNetResNet18"
    
class ProjectorBlockResNet50(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ProjectorBlockResNet50, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.fc(x)

class TrainResNet50():
    def __init__(self, num_classes_downstream=2):

        self.model = torchvision.models.resnet50(weights=None)
        self.criterion = nn.CrossEntropyLoss
        self.optimizer = lambda params: torch.optim.Adam(params, lr=0.001)
        self.num_classes_downstream = num_classes_downstream
        self.mode = "pretext"

    def switch_to_downstream(self):
        if self.mode == "downstream":
            return
            
        self.model.layer3 = nn.Identity()
        self.model.layer4 = nn.Identity()
        
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc = nn.Sequential(
            ProjectorBlockResNet50(512, 256),
            ProjectorBlockResNet50(256, 256),
            ProjectorBlockResNet50(256, self.num_classes_downstream),
            nn.Softmax(dim=1)
        )
        self.mode = "downstream"
        
    def load_weights_from_path(self, path):
        self.model.load_state_dict(torch.load(path))
    
    def __str__(self):
        return "RotNetResNet50"



class TrainResNet18Simple():
    def __init__(self, num_classes_downstream=2):
        self.model = torchvision.models.resnet18(weights=None)
        self.num_classes_downstream = num_classes_downstream

        # The standard ResNet18 has 512 features at the end.
        self.model.fc = nn.Linear(512, self.num_classes_downstream)

        self.criterion = nn.CrossEntropyLoss
        self.optimizer = lambda params: torch.optim.Adam(params, lr=0.001)
        self.mode = "sl" 

    
    def load_weights_from_path(self, path):
        self.model.load_state_dict(torch.load(path))
    
    def __str__(self):
        return "RotNetResNet18Simple"
    
    
class TrainResNet50Simple():
    def __init__(self, num_classes_downstream=2):
        self.model = torchvision.models.resnet50(weights=None)
        self.num_classes_downstream = num_classes_downstream

        # The standard ResNet50 has 2048 features at the end.
        self.model.fc = nn.Linear(2048, self.num_classes_downstream)
        
        self.criterion = nn.CrossEntropyLoss
        self.optimizer = lambda params: torch.optim.Adam(params, lr=0.001)
        self.mode = "sl"
        
    def load_weights_from_path(self, path):
        self.model.load_state_dict(torch.load(path))
    
    def __str__(self):
        return "RotNetResNet50Simple"
