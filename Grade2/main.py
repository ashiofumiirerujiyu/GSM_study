import os
import torch
import torch.nn as nn
import logging
from model import Generator, Discriminator
from torchvision.utils import save_image
from dataloader import get_dataset, get_loaders, get_transform

# 로그 설정
logging.basicConfig(
    filename="/workspace/GSM_study/Grade2/output/training.log",  # 로그 파일 경로
    level=logging.INFO,  # 로그 레벨 설정
    format="%(asctime)s - %(levelname)s - %(message)s",  # 로그 형식 설정
    datefmt="%Y-%m-%d %H:%M:%S",  # 날짜 형식 설정
)


train_transform, test_transform = get_transform()
train_dataset, valid_dataset, test_dataset = get_dataset("/workspace/GSM_study/Grade2/data", 0.8, train_transform, test_transform)
train_loader, valid_loader, test_loader = get_loaders(train_dataset, valid_dataset, test_dataset, 32, 8)


# 손실 함수
loss_func = nn.BCELoss()
n_epoch = 100

# 모델 초기화
latent_dim = 10
generator = Generator(latent_dim)
discriminator = Discriminator()

# 옵티마이저
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0001)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class Trainer:
    def __init__(self, generator, discriminator, optimizer_g, optimizer_d, latent_dim, loss_func, n_epoch, device, save_image_interval, image_save_path):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.latent_dim = latent_dim
        self.loss_func = loss_func
        self.n_epoch = n_epoch
        self.device = device
        self.save_image_interval = save_image_interval
        self.image_save_path = image_save_path

        # Best model tracking
        self.best_g_loss = float('inf')
        self.best_d_loss = float('inf')
        self.best_generator = None
        self.best_discriminator = None

    def main(self, train_loader, valid_loader, test_loader):
        for epoch in range(self.n_epoch):
            train_d_loss, train_g_loss = self.train(train_loader)
            logging.info(f"Train {epoch+1}/{self.n_epoch}:: d_loss: {train_d_loss:.4f} g_loss: {train_g_loss:.4f}")

            valid_d_loss, valid_g_loss = self.valid(valid_loader)
            logging.info(f"Valid {epoch+1}/{self.n_epoch}:: d_loss: {valid_d_loss:.4f} g_loss: {valid_g_loss:.4f}")

            # Check if current model is the best
            if valid_g_loss < self.best_g_loss:
                self.best_g_loss = valid_g_loss
                self.best_d_loss = valid_d_loss
                self.best_generator = self.generator.state_dict()
                self.best_discriminator = self.discriminator.state_dict()
                torch.save(self.best_generator, os.path.join(self.image_save_path, 'best_generator.pth'))
                torch.save(self.best_discriminator, os.path.join(self.image_save_path, 'best_discriminator.pth'))
                logging.info(f"Best model saved at epoch {epoch+1} with valid g_loss: {valid_g_loss:.4f} and d_loss: {valid_d_loss:.4f}")

            # Save generated images at intervals
            if (epoch + 1) % self.save_image_interval == 0:
                self.save_generated_images(epoch + 1)

            # Optionally, you can print the current best model's loss
            logging.info(f"Current Best:: d_loss: {self.best_d_loss:.4f} g_loss: {self.best_g_loss:.4f}")

            if epoch % 100 == 0:
                test_d_loss, test_g_loss = self.test(test_loader)
                logging.info(f"Test {epoch+1}/{self.n_epoch}:: d_loss: {test_d_loss:.4f} g_loss: {test_g_loss:.4f}")

    def train(self, train_loader):
        d_losses = 0.0
        g_losses = 0.0
        for (x, _) in train_loader:
            x = x.to(self.device)

            batch_size = x.size(0)
            real_labels = torch.ones(batch_size, 1).to(self.device)
            fake_labels = torch.zeros(batch_size, 1).to(self.device)

            x = x.view(batch_size, -1)

            # Discriminator 업데이트
            self.optimizer_d.zero_grad()
            real_loss = self.loss_func(self.discriminator(x), real_labels)
            
            z = torch.randn(batch_size, self.latent_dim).to(self.device)
            fake_x = self.generator(z)
            fake_loss = self.loss_func(self.discriminator(fake_x), fake_labels)

            d_loss = real_loss + fake_loss
            d_losses += d_loss.item()
            d_loss.backward()
            self.optimizer_d.step()

            # Generator 업데이트
            self.optimizer_g.zero_grad()
            z = torch.randn(batch_size, self.latent_dim).to(self.device)
            fake_x = self.generator(z)
            g_loss = self.loss_func(self.discriminator(fake_x), real_labels)
            g_losses += g_loss.item()
            g_loss.backward()
            self.optimizer_g.step()

        return d_losses / len(train_loader), g_losses / len(train_loader)
        
    def valid(self, valid_loader):
        d_losses = 0.0
        g_losses = 0.0
        with torch.no_grad():
            for (x, _) in valid_loader:
                x = x.to(self.device)

                batch_size = x.size(0)
                real_labels = torch.ones(batch_size, 1).to(self.device)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)

                x = x.view(batch_size, -1)
                real_loss = self.loss_func(self.discriminator(x), real_labels)

                z = torch.randn(batch_size, self.latent_dim).to(self.device)
                fake_x = self.generator(z)

                fake_loss = self.loss_func(self.discriminator(fake_x), fake_labels)
                d_loss = real_loss + fake_loss
                d_losses += d_loss.item()

                g_loss = self.loss_func(self.discriminator(fake_x), real_labels)
                g_losses += g_loss.item()

        return d_losses / len(valid_loader), g_losses / len(valid_loader)

    def test(self, test_loader):
        d_losses = 0.0
        g_losses = 0.0
        with torch.no_grad():
            for (x, _) in test_loader:
                x = x.to(self.device)

                batch_size = x.size(0)
                real_labels = torch.ones(batch_size, 1).to(self.device)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)

                x = x.view(batch_size, -1)
                real_loss = self.loss_func(self.discriminator(x), real_labels)

                z = torch.randn(batch_size, self.latent_dim).to(self.device)
                fake_x = self.generator(z)

                fake_loss = self.loss_func(self.discriminator(fake_x), fake_labels)
                d_loss = real_loss + fake_loss
                d_losses += d_loss.item()

                g_loss = self.loss_func(self.discriminator(fake_x), real_labels)
                g_losses += g_loss.item()

        return d_losses / len(test_loader), g_losses / len(test_loader)

    def save_generated_images(self, epoch):
        z = torch.randn(64, self.latent_dim).to(self.device)  # 64개의 이미지를 생성
        fake_images = self.generator(z)
        fake_images = fake_images.view(fake_images.size(0), 1, 28, 28)  # 예: MNIST일 경우 28x28 크기로 reshape
        save_image(fake_images, os.path.join(self.image_save_path, f"epoch_{epoch}.png"), nrow=8, normalize=True)

        print(f"Generated images saved for epoch {epoch}")

# Usage example
trainer = Trainer(generator, discriminator, optimizer_G, optimizer_D, latent_dim, loss_func, n_epoch, device, 5, "/workspace/GSM_study/Grade2/output")
trainer.main(train_loader, valid_loader, test_loader)