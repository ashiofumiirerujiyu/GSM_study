import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_dim, label_dim, img_dim, p=0.5):
        super(Generator, self).__init__()
        
        self.label_embedding = nn.Embedding(10, label_dim)
        
        self.model = nn.Sequential(
            nn.Linear(noise_dim + label_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(128, 256),
            nn.InstanceNorm1d(256, momentum=0.8),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(256, 512),
            nn.InstanceNorm1d(512, momentum=0.8),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(512, img_dim),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_input = self.label_embedding(labels)
        gen_input = torch.cat((noise, label_input), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), 1, 28, 28)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_dim, label_dim, p=0.5):
        super(Discriminator, self).__init__()
        
        self.label_embedding = nn.Embedding(10, label_dim)
        
        self.model = nn.Sequential(
            nn.Linear(img_dim + label_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        img_flat = img.view(img.size(0), -1)
        label_input = self.label_embedding(labels)
        d_in = torch.cat((img_flat, label_input), -1)
        validity = self.model(d_in)
        
        return validity
