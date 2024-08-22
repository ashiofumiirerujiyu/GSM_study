import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, noise_dim, label_dim, img_dim, p=0.5):
        super(Generator, self).__init__()
        self.label_embedding = nn.Embedding(10, label_dim)
        self.model = nn.Sequential(
            nn.Linear(noise_dim + label_dim, 128),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(128, 256),
            nn.InstanceNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(256, 512),
            nn.InstanceNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(512, img_dim),
            nn.Tanh()
        )

    def forward(self, noise, label):
        label_input = self.label_embedding(label)
        input = torch.cat((noise, label_input), -1)
        gen_img = self.model(input)
        gen_img = gen_img.view(gen_img.size(0), 1, 28, 28)

        return gen_img
    