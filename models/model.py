import torch.nn as nn
import torchvision.models as models


class Verifier(nn.Module):
    def __init__(self, prenet='resnet18', outdim=128):
        super(Verifier, self).__init__()

        model = getattr(models, prenet)(pretrained=False)

        self.model = list(model.children())[:-1]

        self.backbone = nn.ModuleList()

        for f in self.model:
            self.backbone.append(f)

        # self.backbone = nn.Sequential(*self.model)

        if prenet == 'resnet18':
            nfc = 512
        elif prenet == 'resnet50':
            nfc = 2048

        self.fc1 = nn.Linear(nfc, outdim)
        # self.fc2 = nn.Linear(1024, outdim)

    def forward(self, x):
        bs = x.size(0)

        features = []

        for i, f in enumerate(self.backbone):
            x = f(x)
            if i == 2 or i == 4 or i == 6:
                features.append(x)
        x = x.view(bs, -1)
        output = self.fc1(x)
        return output, features
