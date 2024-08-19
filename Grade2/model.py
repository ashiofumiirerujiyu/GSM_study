import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(True),
            nn.Dropout(0.3),  # 드롭아웃 추가
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Dropout(0.3),  # 드롭아웃 추가
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Dropout(0.3),  # 드롭아웃 추가
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Dropout(0.3),  # 드롭아웃 추가
            nn.Linear(1024, 784),  # 28*28 이미지 크기
            nn.Tanh()  # -1 to 1 범위로 출력
        )

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),  # 드롭아웃 추가
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),  # 드롭아웃 추가
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),  # 드롭아웃 추가
            nn.Linear(128, 1),
            nn.Sigmoid()  # 출력은 0~1 사이의 확률 값
        )

    def forward(self, img):
        return self.model(img)


class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(True),
            nn.Dropout(0.3),  # 드롭아웃 추가
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Dropout(0.3),  # 드롭아웃 추가
            nn.Linear(128, num_classes)
        )

    def forward(self, z):
        return self.model(z)


# class Classifier(nn.Module):
#     def __init__(self, input_dim, num_classes):
#         super(Classifier, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(input_dim, num_classes)
#         )

#     def forward(self, z):
#         return self.model(z)