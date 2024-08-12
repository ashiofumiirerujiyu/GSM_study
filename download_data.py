import torch
import torchvision
from torchvision import transforms
from PIL import Image
import os

# 'data' 디렉토리 생성 (존재하지 않을 경우)
data_dir = './data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# CIFAR-10 데이터셋 로드
transform = transforms.Compose([transforms.ToTensor()])
dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)

# 레이블 이름
class_names = dataset.classes

# 'dog'와 'cat'의 인덱스
dog_class_index = class_names.index('dog')
cat_class_index = class_names.index('cat')

# 저장할 디렉토리 생성
save_dir = os.path.join(data_dir, 'cifar10_images')
os.makedirs(save_dir, exist_ok=True)
dog_dir = os.path.join(save_dir, 'dog')
cat_dir = os.path.join(save_dir, 'cat')
os.makedirs(dog_dir, exist_ok=True)
os.makedirs(cat_dir, exist_ok=True)

# 'dog'와 'cat' 이미지 각각 100장 저장
def save_images(class_index, save_path, num_images=500):
    count = 0
    for img, label in dataset:
        if count >= num_images:
            break
        if label == class_index:
            img = transforms.ToPILImage()(img)
            img.save(os.path.join(save_path, f'{count}.png'))
            count += 1

# 'dog'와 'cat' 이미지 저장
save_images(dog_class_index, dog_dir)
save_images(cat_class_index, cat_dir)

print('Images have been saved successfully.')
