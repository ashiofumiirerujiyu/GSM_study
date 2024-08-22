import os
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image


class Trainer:
    def __init__(self, generator, discriminator, optimizer_g, optimizer_d, loss_func, noise_dim, n_epoch, device, save_interval, save_path, logger):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.loss_func = loss_func
        self.noise_dim = noise_dim
        self.n_epoch = n_epoch
        self.device = device
        self.save_interval = save_interval
        self.save_path = save_path
        self.logger = logger

        self.writer = SummaryWriter(log_dir=os.path.join(save_path))

        self.best_g_loss = float('inf')
        self.best_d_loss = float('inf')
        self.best_generator = None
        self.best_discriminator = None
        self.best_classifier = None

    def main(self, train_loader, valid_loader, test_loader):
        for epoch in range(self.n_epoch):
            train_d_loss, train_g_loss = self.train(train_loader)
            self.logger.info(f"Train {epoch+1}/{self.n_epoch}:: d_loss: {train_d_loss:.4f} g_loss: {train_g_loss:.4f}")

            valid_d_loss, valid_g_loss = self.valid(valid_loader)
            self.logger.info(f"Valid {epoch+1}/{self.n_epoch}:: d_loss: {valid_d_loss:.4f} g_loss: {valid_g_loss:.4f}")

            self.writer.add_scalars('Losses', {
                'Train_D_Loss': train_d_loss,
                'Train_G_Loss': train_g_loss,
                'Valid_D_Loss': valid_d_loss,
                'Valid_G_Loss': valid_g_loss
            }, epoch)

            if valid_g_loss < self.best_g_loss:
                self.best_g_loss = valid_g_loss
                self.best_d_loss = valid_d_loss
                self.best_generator = self.generator.state_dict()
                self.best_discriminator = self.discriminator.state_dict()
                torch.save(self.best_generator, os.path.join(self.save_path, 'best_generator.pth'))
                torch.save(self.best_discriminator, os.path.join(self.save_path, 'best_discriminator.pth'))
                self.logger.info(f"Best model saved at epoch {epoch+1} with valid g_loss: {valid_g_loss:.4f}, d_loss: {valid_d_loss:.4f},")

            if (epoch + 1) % self.save_interval == 0:
                for i in range(10):
                    random_label = torch.tensor(i)
                    self.save_generated_images(epoch + 1, random_label)

            self.logger.info(f"Current Best:: d_loss: {self.best_d_loss:.4f} g_loss: {self.best_g_loss:.4f}")

            if epoch % 100 == 0:
                test_d_loss, test_g_loss = self.test(test_loader)
                self.logger.info(f"Test {epoch+1}/{self.n_epoch}:: d_loss: {test_d_loss:.4f} g_loss: {test_g_loss:.4f}")
                self.writer.add_scalars('Test_Losses', {
                    'Test_D_Loss': test_d_loss,
                    'Test_G_Loss': test_g_loss
                }, epoch)

    def train(self, train_loader):
        d_losses = 0.0
        g_losses = 0.0
        for (x, y) in train_loader:
            x, y = x.to(self.device), y.to(self.device)

            batch_size = x.size(0)
            real_labels = torch.ones(batch_size, 1).to(self.device)
            fake_labels = torch.zeros(batch_size, 1).to(self.device)

            x = x.view(batch_size, -1)

            self.optimizer_d.zero_grad()
            real_loss = self.loss_func(self.discriminator(x, y), real_labels)
            
            z = torch.randn(batch_size, self.noise_dim).to(self.device)
            fake_x = self.generator(z, y)
            fake_loss = self.loss_func(self.discriminator(fake_x, y), fake_labels)

            d_loss = real_loss + fake_loss
            d_losses += d_loss.item()
            d_loss.backward()
            self.optimizer_d.step()

            self.optimizer_g.zero_grad()
            z = torch.randn(batch_size, self.noise_dim).to(self.device)
            fake_x = self.generator(z, y)
            g_loss = self.loss_func(self.discriminator(fake_x, y), real_labels)
            g_losses += g_loss.item()
            g_loss.backward()
            self.optimizer_g.step()

        return d_losses / len(train_loader), g_losses / len(train_loader)
        
    def valid(self, valid_loader):
        d_losses = 0.0
        g_losses = 0.0
        with torch.no_grad():
            for (x, y) in valid_loader:
                x, y = x.to(self.device), y.to(self.device)

                batch_size = x.size(0)
                real_labels = torch.ones(batch_size, 1).to(self.device)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)

                x = x.view(batch_size, -1)
                real_loss = self.loss_func(self.discriminator(x, y), real_labels)

                z = torch.randn(batch_size, self.noise_dim).to(self.device)
                fake_x = self.generator(z, y)

                fake_loss = self.loss_func(self.discriminator(fake_x, y), fake_labels)
                d_loss = real_loss + fake_loss
                d_losses += d_loss.item()

                g_loss = self.loss_func(self.discriminator(fake_x, y), real_labels)
                g_losses += g_loss.item()

        return d_losses / len(valid_loader), g_losses / len(valid_loader)

    def test(self, test_loader):
        d_losses = 0.0
        g_losses = 0.0
        with torch.no_grad():
            for (x, y) in test_loader:
                x, y = x.to(self.device), y.to(self.device)

                batch_size = x.size(0)
                real_labels = torch.ones(batch_size, 1).to(self.device)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)

                x = x.view(batch_size, -1)
                real_loss = self.loss_func(self.discriminator(x, y), real_labels)

                z = torch.randn(batch_size, self.noise_dim).to(self.device)
                fake_x = self.generator(z, y)

                fake_loss = self.loss_func(self.discriminator(fake_x, y), fake_labels)
                d_loss = real_loss + fake_loss
                d_losses += d_loss.item()

                g_loss = self.loss_func(self.discriminator(fake_x, y), real_labels)
                g_losses += g_loss.item()

        return d_losses / len(test_loader), g_losses / len(test_loader)

    def save_generated_images(self, epoch, y):
        y = y.unsqueeze(0).to(self.device)
        z = torch.randn(1, self.noise_dim).to(self.device)
        
        fake_image = self.generator(z, y)
        fake_image = fake_image.view(1, 1, 28, 28)
        
        filename = os.path.join(self.save_path, f"epoch_{epoch}_{int(y.detach())}.jpg")
        save_image(fake_image, filename, nrow=1, normalize=True)

        self.logger.info(f"Generated image saved for epoch {epoch} as {filename}")