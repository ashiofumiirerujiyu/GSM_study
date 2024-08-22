import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, img_dim, label_dim, p=0.5):
        super(Discriminator, self).__init__()
        self.label_embedding = nn.Embedding(10, label_dim)
        self.model = nn.Sequential(
            nn.Linear(img_dim + label_dim, 512),
            nn.LeakyReLU(),
            nn.Dropout(p),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Dropout(p),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        img_flat = img.view(img.size(0), -1)
        label_input = self.label_embedding(labels)
        dis_input = torch.cat((img_flat, label_input), -1)
        validity = self.model(dis_input)
        
        return validity
    