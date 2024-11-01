import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from time import time
from typing import List, Tuple

from conv2d_mine import CNN
from conv2d_torch import SimpleCNN, ResNet
from data_handler import train_data, train_loader, val_loader, small_train_loader, classes
from model import ResNet as SC

# Check if GPU is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def train(
    model: nn.Module, 
    train_dataset: torch.utils.data.DataLoader, 
    val_dataset: torch.utils.data.DataLoader,
    criterion: nn.Module, 
    optimizer = torch.optim, 
    epochs: int = 10,
    lr: float = 0.001,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    model.train()
    model.to(device)
    train_loss_history, train_accuracy_history, val_accuracy_history, val_loss_history = [], [], [], []
    optimizer = optimizer(model.parameters(), lr=lr)
    
    for epoch in range(1, 1 + epochs):
        epoch_loss = 0
        correct = 0
        total = 0
        begin = time()
        
        # Training phase
        for step, (batch_x, batch_y) in enumerate(train_dataset):
            # begin_step = time()
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            out = model(batch_x)
            loss = criterion(out, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = torch.max(out, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
            
            # end_step = time()
            # Step별 손실 출력
            # print(f'[epoch {epoch}/{epochs} - step {step + 1}/{len(train_dataset)}] step loss: {round(loss.item(), 4)} time_passed: {round(end_step - begin_step, 4)}')
        
        accuracy = correct / total
        train_loss_history.append(epoch_loss / len(train_dataset))
        train_accuracy_history.append(accuracy)
        
        # Validation phase
        val_accuracy, val_loss = evaluate(model, val_dataset, criterion)
        val_accuracy_history.append(val_accuracy)
        val_loss_history.append(val_loss)

        end = time()
        print(f'[epoch {epoch}/{epochs}] train loss : {round(epoch_loss / len(train_dataset), 4)}, train accuracy: {round(accuracy * 100, 2)}%, validation loss: {round(val_loss, 4)}, validation accuracy: {round(val_accuracy * 100, 2)}%, time: {round(end - begin, 4)} sec passed.')
        
        save_model_path = f'saved_models/model_epoch_{epoch}.pth'
        os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
        torch.save(model.state_dict(), save_model_path)
        print(f'Model saved to {save_model_path}')

    return train_loss_history, train_accuracy_history, val_accuracy_history, val_loss_history

def evaluate(model: nn.Module, 
             val_loader: torch.utils.data.DataLoader, 
             criterion: nn.Module) -> Tuple[float, float]:
    model.eval()  # 평가 모드 설정
    model.to(device)
    
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():  # 평가 중에는 gradient 계산 비활성화
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            out = model(batch_x)
            
            loss = criterion(out, batch_y)
            val_loss += loss.item()
            
            _, predicted = torch.max(out, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
    
    accuracy = correct / total
    avg_val_loss = val_loss / len(val_loader)
    print(f'Evaluation accuracy: {round(accuracy * 100, 2)}%, loss: {round(avg_val_loss, 4)}')
    
    return accuracy, avg_val_loss

def plot_metrics(
    loss_history: List[float], 
    accuracy_history: List[float], 
    val_accuracy_history: List[float], 
    val_loss_history: List[float], 
    save_path: str = None
):
    plt.figure(figsize=(12, 5))
    
    # Training & Validation Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(loss_history, label='Training Loss')
    plt.plot(val_loss_history, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss History')
    plt.legend()
    plt.grid(True)
    
    # Training & Validation Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(accuracy_history, label='Training Accuracy', color='orange')
    plt.plot(val_accuracy_history, label='Validation Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy History')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path) 
        print(f"Plot saved to {save_path}")
        
    plt.show()

    
if __name__ == '__main__':
    ## My_CNN
    # model = CNN(train_data.classes, device).to(device)
    # train_loss_history, train_accuracy_history, val_accuracy_history = train(
    #     model, 
    #     train_loader, 
    #     val_loader, 
    #     criterion=nn.CrossEntropyLoss(), 
    #     optimizer=optim.Adam, 
    #     epochs=10, 
    #     lr=0.001, 
    # )

    # # 훈련 데이터셋에서 테스트 정확도 평가 (이 부분은 선택적)
    # # test_accuracy = evaluate(model, train_loader)
    
    # plot_metrics(train_loss_history, train_accuracy_history, val_accuracy_history)


    ## CNN
    # model = SimpleCNN(len(classes), device).to(device)
    # train_loss_history, train_accuracy_history, val_accuracy_history, val_loss_history = train(
    #     model, 
    #     train_loader, 
    #     val_loader,
    #     criterion=nn.CrossEntropyLoss(), 
    #     optimizer=optim.Adam, 
    #     epochs = 25, 
    #     lr = 0.002,
    # )
    
    # plot_metrics(train_loss_history, train_accuracy_history, val_accuracy_history, val_loss_history, save_path = "saved_plot/plot.png")
    
    
    ## ResNet
    model = ResNet(3, 16, len(classes), device = device).to(device)
    train_loss_history, train_accuracy_history, val_accuracy_history, val_loss_history = train(
        model, 
        train_loader, 
        val_loader,
        criterion=nn.CrossEntropyLoss(), 
        optimizer=optim.Adam, 
        epochs = 25, 
        lr = 0.002,
    )
    
    plot_metrics(train_loss_history, train_accuracy_history, val_accuracy_history, val_loss_history, save_path = "saved_plot/plot.png")
    
    ## SC
        # conv2d_torch.ResNet
    # model = SC(3, 16, len(classes), device=device).to(device)
    # train_loss_history, train_accuracy_history, valid_loss_history, valid_accuracy_history = train(
    #     model,
    #     # small_train_loader,
    #     train_loader,
    #     val_loader,
    #     criterion=nn.CrossEntropyLoss(),
    #     optimizer=optim.Adam,
    #     epochs = 30,
    #     lr = 0.001,
    # )
    
    # plot_metrics(train_loss_history, train_accuracy_history, valid_loss_history, valid_accuracy_history, save_path = "saved_plot/plot.png")