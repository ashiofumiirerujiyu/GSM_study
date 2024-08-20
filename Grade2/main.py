import os
import random
import torch
import torch.nn as nn
import logging
from model import Generator, Discriminator
from torchvision.utils import save_image
from dataloader import get_dataset, get_loaders, get_transform
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import pytz

# 한국 표준시(KST) 설정
kst = pytz.timezone('Asia/Seoul')

class KSTFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        # 로그 기록의 시간대를 한국 표준시로 변환
        log_time = datetime.fromtimestamp(record.created, kst)
        if datefmt:
            return log_time.strftime(datefmt)
        else:
            return log_time.strftime('%Y-%m-%d %H:%M:%S')

def set_seed(seed):
    """
    Sets the seed for random number generation to ensure reproducibility.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_save_path(base_path, seed):
    """
    Generates a unique save path based on the current date, time, and seed.
    """
    now = datetime.now(kst)
    date_str = now.strftime("%Y_%m_%d")
    time_str = now.strftime("%H_%M")
    path = os.path.join(base_path, f"{date_str}", f"{time_str}_{seed}")
    os.makedirs(path, exist_ok=True)
    return path

# Set a seed for reproducibility
seed = 42
set_seed(seed)  # You can use any integer here

script_path = os.path.abspath(__file__)
script_directory = os.path.dirname(script_path)

# Generate image save path
base_output_path = os.path.join(script_directory, "output")
if not os.path.exists(base_output_path):
    os.makedirs(base_output_path, exist_ok=True)
image_save_path = generate_save_path(base_output_path, seed)

# 로그 설정
formatter = KSTFormatter('%(asctime)s - %(levelname)s - %(message)s')

# 파일 핸들러 설정
file_handler = logging.FileHandler(os.path.join(image_save_path, "training.log"))
file_handler.setFormatter(formatter)

# 로거 설정
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)

# 테스트 로그
logger.info("Logging setup complete.")


train_transform, test_transform = get_transform()
train_dataset, valid_dataset, test_dataset = get_dataset("/workspace/GSM_study/Grade2/data", 0.8, train_transform, test_transform)
train_loader, valid_loader, test_loader = get_loaders(train_dataset, valid_dataset, test_dataset, 128, 4)

# 손실 함수
loss_func = nn.BCELoss()
cls_loss_func = nn.CrossEntropyLoss()
n_epoch = 1000

# 모델 초기화
noise_dim = 100
label_dim = 10
img_dim = 784
generator = Generator(noise_dim, label_dim, img_dim)
discriminator = Discriminator(img_dim, label_dim)

# 옵티마이저
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0001)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")     
print(f"Using device: {device}")

class Trainer:
    def __init__(self, generator, discriminator, optimizer_g, optimizer_d, loss_func, noise_dim, n_epoch, device, save_image_interval, image_save_path):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.loss_func = loss_func
        self.noise_dim = noise_dim
        self.n_epoch = n_epoch
        self.device = device
        self.save_image_interval = save_image_interval
        self.image_save_path = image_save_path

        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=os.path.join(image_save_path))

        # Best model tracking
        self.best_g_loss = float('inf')
        self.best_d_loss = float('inf')
        self.best_generator = None
        self.best_discriminator = None
        self.best_classifier = None

    def main(self, train_loader, valid_loader, test_loader):
        for epoch in range(self.n_epoch):
            train_d_loss, train_g_loss = self.train(train_loader)
            logging.info(f"Train {epoch+1}/{self.n_epoch}:: d_loss: {train_d_loss:.4f} g_loss: {train_g_loss:.4f}")

            valid_d_loss, valid_g_loss = self.valid(valid_loader)
            logging.info(f"Valid {epoch+1}/{self.n_epoch}:: d_loss: {valid_d_loss:.4f} g_loss: {valid_g_loss:.4f}")

            # Log losses to TensorBoard
            self.writer.add_scalars('Losses', {
                'Train_D_Loss': train_d_loss,
                'Train_G_Loss': train_g_loss,
                'Valid_D_Loss': valid_d_loss,
                'Valid_G_Loss': valid_g_loss
            }, epoch)

            # Check if current model is the best
            if valid_g_loss < self.best_g_loss:
                self.best_g_loss = valid_g_loss
                self.best_d_loss = valid_d_loss
                self.best_generator = self.generator.state_dict()
                self.best_discriminator = self.discriminator.state_dict()
                torch.save(self.best_generator, os.path.join(self.image_save_path, 'best_generator.pth'))
                torch.save(self.best_discriminator, os.path.join(self.image_save_path, 'best_discriminator.pth'))
                logging.info(f"Best model saved at epoch {epoch+1} with valid g_loss: {valid_g_loss:.4f}, d_loss: {valid_d_loss:.4f},")

            # Save generated images at intervals
            if (epoch + 1) % self.save_image_interval == 0:
                random_label = torch.tensor(random.randint(0, 9))
                self.save_generated_images(epoch + 1, random_label)

            # Optionally, you can print the current best model's loss
            logging.info(f"Current Best:: d_loss: {self.best_d_loss:.4f} g_loss: {self.best_g_loss:.4f}")

            if epoch % 100 == 0:
                test_d_loss, test_g_loss = self.test(test_loader)
                logging.info(f"Test {epoch+1}/{self.n_epoch}:: d_loss: {test_d_loss:.4f} g_loss: {test_g_loss:.4f}")

                # Log test losses to TensorBoard
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

            # Discriminator 업데이트
            self.optimizer_d.zero_grad()
            real_loss = self.loss_func(self.discriminator(x, y), real_labels)
            
            z = torch.randn(batch_size, self.noise_dim).to(self.device)
            fake_x = self.generator(z, y)
            fake_loss = self.loss_func(self.discriminator(fake_x, y), fake_labels)

            d_loss = real_loss + fake_loss
            d_losses += d_loss.item()
            d_loss.backward()
            self.optimizer_d.step()

            # Generator 업데이트
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
        # 1개의 노이즈 벡터 생성
        y = y.unsqueeze(0).to(self.device)
        z = torch.randn(1, self.noise_dim).to(self.device)
        
        fake_image = self.generator(z, y)
        
        # 이미지 크기 변환 (예: MNIST일 경우 28x28 크기로 reshape)
        fake_image = fake_image.view(1, 1, 28, 28)
        
        # 파일 이름에 클래스를 포함하여 저장
        filename = os.path.join(self.image_save_path, f"epoch_{epoch}_{int(y.detach())}.png")
        save_image(fake_image, filename, nrow=1, normalize=True)

        print(f"Generated image saved for epoch {epoch} as {filename}")

# Usage example
trainer = Trainer(generator, discriminator, optimizer_G, optimizer_D, loss_func, noise_dim, n_epoch, device, 5, image_save_path)
trainer.main(train_loader, valid_loader, test_loader)
