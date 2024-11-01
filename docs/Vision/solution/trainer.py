import torch 
import torch.nn as nn 
import torch.optim as optim
import matplotlib.pyplot as plt
from time import time 
from typing import List 

from conv2d_mine import CNN 
from conv2d_torch import SimpleCNN, ResNet 
from data_handler import train_data, train_loader, small_train_loader

def train(
    model: nn.Module, 
    dataset: torch.utils.data.DataLoader, 
    criterion: nn.Module, 
    optimizer: torch.optim, 
    epochs: int = 10, 
    lr: float = 0.001, 
) -> List[float]:
    model.train()
    train_loss_history, train_accuray = [], []
    optimizer = optimizer(model.parameters(), lr = lr)
    for epoch in range(1, 1 + epochs):
        epoch_loss = 0
        no_batches = 0
        begin = time()
        
        for batch_x, batch_y in dataset:
            out = model(batch_x) 
            loss = criterion(out, batch_y)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item() 
            no_batches += 1 
        end = time() 
        train_loss_history.append(epoch_loss / no_batches)
        print(f'[epoch {epoch}/epochs] train loss : {round(epoch_loss / no_batches, 4)}, {round(end - begin, 4)} sec passed.')

    return train_loss_history

def plot_loss_history(loss_history):
    """
    Plots the training and validation loss history.

    Parameters:
    - loss_history (dict): A dictionary containing 'train' and 'val' loss lists.

    Example usage:
    loss_history = {'train': [0.9, 0.7, 0.5], 'val': [1.0, 0.8, 0.6]}
    plot_loss_history(loss_history)
    """
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label='Training Loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss History')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # conv2d_mine 
    # train_loss_history = train(
    #     CNN(train_data.classes) , 
    #     small_train_loader, 
    #     criterion = nn.CrossEntropyLoss(), 
    #     optimizer = optim.Adam, 
    #     epochs = 10, 
    #     lr = 0.001, 
    # )

    # plot_loss_history(train_loss_history)
    
    # conv2d_torch.SimpleCNN 
    # train_loss_history = train(
    #     SimpleCNN(3, 16, len(train_data.classes)), 
    #     small_train_loader, 
    #     criterion = nn.CrossEntropyLoss(), 
    #     optimizer = optim.Adam, 
    #     epochs = 10, 
    #     lr = 0.001, 
    # )

    # plot_loss_history(train_loss_history)

    # conv2d_torch.ResNet
    train_loss_history = train(
        ResNet(3, 16, len(train_data.classes)), 
        small_train_loader, 
        criterion = nn.CrossEntropyLoss(), 
        optimizer = optim.Adam, 
        epochs = 10, 
        lr = 0.001, 
    )

    plot_loss_history(train_loss_history)