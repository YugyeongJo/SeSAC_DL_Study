import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import config
from torch.utils.data import ConcatDataset, DataLoader, random_split

transform = transforms.Compose([transforms.ToTensor()])
transform_flip = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 좌우 반전
    transforms.ToTensor()
])

train_data = torchvision.datasets.CIFAR10(root = './data', train = True, download = False, transform = transform)
classes = train_data.classes 
train_data_flip = torchvision.datasets.CIFAR10(root = './data', train = True, download = False, transform = transform_flip)

train_data = ConcatDataset([train_data, train_data_flip])

train_size = int(0.8 * len(train_data))  # 80%
val_size = len(train_data) - train_size  # 나머지 20%

train_dataset, val_dataset = random_split(train_data, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

small_train_loader = []
small_dataset_size = 10
size = 0

for batch_x, batch_y in train_loader:
    size += 1
    small_train_loader.append((batch_x, batch_y))
    if size >= small_dataset_size:
        break

if __name__ == '__main__':
    print(len(train_data))