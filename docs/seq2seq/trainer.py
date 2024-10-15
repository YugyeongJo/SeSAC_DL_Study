import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os

from torch.utils.data import DataLoader
from data_handler import Vocabulary

class Trainer:
    """_summary_
    """
    
    def __init__(self, model, device, criterion, optimizer, max_target_len = 100, save_dir = 'models'):
        """_summary_

        Args:
            model (_type_): _description_
            device (_type_): _description_
            criterion (_type_): _description_
            optimizer (_type_): _description_
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok = True)
        self.max_target_len = max_target_len
        
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
    def train(self, train_loader, valid_loader=None, num_epochs=10, print_every=1, evaluate_every=1, evaluate_metric = 'accuracy'):
        """_summary_
        """
        val_counter = 0
        for epoch in range(1, num_epochs+1):
            self.model.train()
            epoch_train_loss = 0
            
            for batch_idx, (source, target) in enumerate(train_loader):
                source = source.to(self.device)
                target = target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(source, target)
                
                output_dim = output.shape[-1]
                output = output.reshape(-1, output_dim)
                target = target.reshape(-1)
                
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                
                epoch_train_loss += loss.item()
                
        avg_train_loss = epoch_train_loss / len(train_loader)
        self.train_losses.append(avg_train_loss)
        
        if epoch % print_every == 0:
            print(f'Epoch {epoch}/{num_epochs}, Training Loss: {avg_train_loss:.4f}')
            
        if valid_loader and epoch % evaluate_every == 0:
            val_loss, val_metric = self.evaluate(valid_loader, evaluate_metric=evaluate_metric)
            self.val_losses.append(val_loss)
            self.val_metrics.append(val_metric)
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict()
                torch.save(self.best_model_state, os.path.join(self.save_dir, 'best_model.pth'))
                print(f'Best model saved with validataion loss: {val_loss:.4f}, validation {evaluate_metric}: {val_metric:.4f}')
                val_counter = 0
            else:
                val_counter += 1
                
        if val_counter > 10:
            self.plot_losses()
            return 
            
    def evaluate(self, valid_loader, evaluate_metric = 'accuracy'):
        """_summary_
        """
        self.model.eval()
        epoch_val_loss = 0
        
        if evaluate_metric == 'accuracy':
            correct = 0
            total = 0
        
        with torch.no_grad():
            for source, target in valid_loader:
                source = source.to(self.device)
                target = target.to(self.device)
                
                output = self.model(source, target, teacher_forcing_ratio=0)
                output_dim = output.shape[-1]
                
                if evaluate_metric == 'accuracy':
                    output = output[:, 1:].reshape(-1, output_dim)
                    target = target[:, 1:].reshape(-1)
                    pred = torch.argmax(output, dim = -1)
                    correct += torch.sum((pred == target).float())
                    total += target.numel()
                
                loss = self.criterion(output, target)
                epoch_val_loss += loss.item()
                
        if evaluate_metric == 'accuracy':
            metric = correct / total
            
        avg_val_loss = epoch_val_loss / len(valid_loader)
        print(f'Validation Loss: {avg_val_loss:.4f}, Validation {evaluate_metric}:{metric:.4f}')
        
        return avg_val_loss, metric 
        