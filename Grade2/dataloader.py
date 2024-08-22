import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split


def get_transform(logger):
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    logger.info(f"train transform: {train_transform}")
    logger.info(f"test transform: {test_transform}")

    return train_transform, test_transform


def get_dataset(data_dir, train_ratio, train_transform, test_transform, logger):
    train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True, transform=test_transform)
    test_dataset = torchvision.datasets.MNIST(data_dir, train=False, download=True, transform=test_transform)

    train_size = int(train_ratio * len(train_dataset))
    valid_size = len(train_dataset) - train_size

    train_dataset, valid_dataset = random_split(train_dataset, [train_size, valid_size])
    train_dataset.dataset.transform = train_transform

    logger.info(f"length of train dataset: {len(train_dataset)}")
    logger.info(f"length of valid dataset: {len(valid_dataset)}")
    logger.info(f"length of test dataset: {len(test_dataset)}")

    return train_dataset, valid_dataset, test_dataset


def get_loaders(train_dataset, valid_dataset, test_dataset, batch_size, num_workers, logger):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    logger.info(f"length of train loader: {len(train_loader)}")
    logger.info(f"length of valid loader: {len(valid_loader)}")
    logger.info(f"length of test loader: {len(test_loader)}")

    return train_loader, valid_loader, test_loader
