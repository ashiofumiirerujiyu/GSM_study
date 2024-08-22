import os
import logging
import yaml
import torch
import torch.nn as nn
from dataloader import get_transform, get_dataset, get_loaders
from utils import KSTFormatter, set_seed, generate_save_path
from model import Generator, Discriminator
from trainer import Trainer

yaml_path = "/workspace/GSM_study/Grade2/hyperparameter.yaml"
with open(yaml_path, 'r') as file:
    config = yaml.safe_load(file)

# seed fix
set_seed(config['train']['seed'])

# save path
save_path = generate_save_path(config['train']['seed'])

# save yaml
with open(os.path.join(save_path, 'config.yaml'), 'w') as file:
    yaml.dump(config, file)

# logger
formatter = KSTFormatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler(os.path.join(save_path, "training.log"))
file_handler.setFormatter(formatter)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# dataset
train_transform, test_transform = get_transform(logger)
train_dataset, valid_dataset, test_dataset = get_dataset("/workspace/GSM_study/Grade2/data", config['dataloader']['train_ratio'], train_transform, test_transform, logger)
train_loader, valid_loader, test_loader = get_loaders(train_dataset, valid_dataset, test_dataset, config['dataloader']['batch_size'], config['dataloader']['num_workers'], logger)

# model
generator = Generator(config['model']['noise_dim'], config['model']['label_dim'], config['model']['img_dim'])
discriminator = Discriminator(config['model']['img_dim'], config['model']['label_dim'])

# trainer
loss_func = nn.BCELoss()
optimizer_G = torch.optim.Adam(generator.parameters(), lr=config['train']['gen_lr'])
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=config['train']['dis_lr'])

trainer = Trainer(generator, discriminator, optimizer_G, optimizer_D, loss_func, config['model']['noise_dim'], config['train']['n_epoch'], device, config['train']['save_interval'], save_path, logger)
trainer.main(train_loader, valid_loader, test_loader)
