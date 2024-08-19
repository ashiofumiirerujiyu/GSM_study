import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split


def get_transform():
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # 이미지를 좌우로 뒤집기
        transforms.RandomRotation(10),      # 이미지를 -10도에서 10도 사이로 회전
        transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),  # 이미지의 일부를 크롭하고 다시 28x28 크기로 조정
        transforms.ToTensor(),              # 이미지를 텐서로 변환
        transforms.Normalize((0.5,), (0.5,)),  # 픽셀 값을 [-1, 1]로 정규화
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),              # 이미지를 텐서로 변환
        transforms.Normalize((0.5,), (0.5,)),  # 픽셀 값을 [-1, 1]로 정규화
    ])

    if __debug__:
        print(f"train transform: {train_transform}")
        print(f"test transform: {test_transform}")

    return train_transform, test_transform


def get_dataset(data_dir, train_ratio, train_transform, test_transform):
    train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True, transform=test_transform)
    test_dataset = torchvision.datasets.MNIST(data_dir, train=False, download=True, transform=test_transform)

    train_size = int(train_ratio * len(train_dataset))
    valid_size = len(train_dataset) - train_size

    train_dataset, valid_dataset = random_split(train_dataset, [train_size, valid_size])
    train_dataset.dataset.transform = train_transform

    if __debug__:
        print(f"length of train dataset: {len(train_dataset)}")
        print(f"length of valid dataset: {len(valid_dataset)}")
        print(f"length of test dataset: {len(test_dataset)}")

    return train_dataset, valid_dataset, test_dataset


def get_loaders(train_dataset, valid_dataset, test_dataset, batch_size, num_workers):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    if __debug__:
        print(f"length of train loader: {len(train_loader)}")
        print(f"length of valid loader: {len(valid_loader)}")
        print(f"length of test loader: {len(test_loader)}")

    return train_loader, valid_loader, test_loader
