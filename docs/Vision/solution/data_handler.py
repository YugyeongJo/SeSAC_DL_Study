import torch 
import torch.nn as nn 
import torchvision 
import torchvision.transforms as transforms 
from torch.utils.data import DataLoader

import config 
from debugger import debug_shell

transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
])

train_data = torchvision.datasets.CIFAR10(root = '../data', train = True, download = False, transform=transform)

train_loader = DataLoader(train_data, batch_size = config.batch_size, shuffle = True) 

small_train_loader = []
small_dataset_size = 10
size = 0

for batch_x, batch_y in train_loader:
    size += 1 
    small_train_loader.append((batch_x, batch_y))
    if size < small_dataset_size:
        break 
