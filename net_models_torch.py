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
    def __init__(self):
        self.model = torchvision.models.resnet18()
        self.criterion = nn.CrossEntropyLoss
        self.optimizer = lambda params: torch.optim.Adam(params, lr=0.001)
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
            ProjectorBlockResNet18(256, 10),
            nn.Softmax(dim=1)
        )

        self.mode = "downstream"

        print("Switched model to downstream mode")


    def load_weights_from_path(self, path):
        self.model.load_state_dict(torch.load(path))

    def __str__(self):
        return "RotNetResNet18"
    
