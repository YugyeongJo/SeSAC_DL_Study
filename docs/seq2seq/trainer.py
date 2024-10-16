import os
import random 
import torch 
import torch.nn as nn 
import matplotlib.pyplot as plt

import os
import random 
import torch 
import torch.nn as nn 

class Seq2SeqTrainer:
    def __init__(self, model, train_loader, valid_loader, optimizer, criterion, device):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
        # 저장할 폴더 경로 설정
        self.save_dir = 'saved_models'
        os.makedirs(self.save_dir, exist_ok=True)  # 폴더가 없으면 생성
        self.save_path = os.path.join(self.save_dir, 'best_model.pth')  # 모델 저장 경로 설정
        
        self.train_losses = []
        self.train_accuracies = []
        self.best_loss = float('inf')  # Initialize best_loss as infinity

    def train(self, num_epochs):
        self.model.train()
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            correct_predictions = 0
            total_predictions = 0
            
            for source, target in self.train_loader:
                source, target = source.to(self.device), target.to(self.device)  # Move data to device
                self.optimizer.zero_grad()  # Zero the gradients
                
                outputs = self.model(source, target[:, :-1])  # Forward pass (excluding last target)
                loss = self.criterion(outputs.view(-1, outputs.size(-1)), target[:, 1:].view(-1))  # Compute loss
                loss.backward()  # Backpropagation
                self.optimizer.step()  # Update weights
                
                epoch_loss += loss.item()  # Accumulate loss
                
                # Calculate accuracy
                _, predicted = torch.max(outputs, dim=-1)  # Get predicted classes
                correct_predictions += (predicted == target[:, 1:]).sum().item()  # Count correct predictions
                total_predictions += target[:, 1:].numel()  # Total predictions made
            
            # Calculate average loss and accuracy for the epoch
            avg_loss = epoch_loss / len(self.train_loader)
            accuracy = correct_predictions / total_predictions
            self.train_losses.append(avg_loss)
            self.train_accuracies.append(accuracy)
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Train Accuracy: {accuracy:.4f}')
            
            # Evaluate on validation set
            valid_loss, valid_accuracy = self.evaluate()
            print(f'Epoch [{epoch+1}/{num_epochs}], Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.4f}')
            
            # Save the best model based on the validation loss
            if valid_loss < self.best_loss:
                self.best_loss = valid_loss
                torch.save(self.model.state_dict(), self.save_path)  # Save model state
                print(f'Saved best model with valid loss: {self.best_loss:.4f} to {self.save_path}')
                # Plot and save the loss and accuracy when saving the best model
                self.plot()

    def evaluate(self):
        self.model.eval()
        valid_loss = 0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for source, target in self.valid_loader:
                source, target = source.to(self.device), target.to(self.device)
                outputs = self.model(source, target[:, :-1])
                loss = self.criterion(outputs.view(-1, outputs.size(-1)), target[:, 1:].view(-1))
                valid_loss += loss.item()
                
                _, predicted = torch.max(outputs, dim=-1)
                correct_predictions += (predicted == target[:, 1:]).sum().item()
                total_predictions += target[:, 1:].numel()
        
        avg_valid_loss = valid_loss / len(self.valid_loader)
        accuracy = correct_predictions / total_predictions
        return avg_valid_loss, accuracy

    def plot(self):
        # Create a figure for loss and accuracy
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Plotting the loss
        ax1.plot(self.train_losses, label='Training Loss', color='blue')
        ax1.set_title('Training Loss Over Epochs')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Plotting the accuracy
        ax2.plot(self.train_accuracies, label='Training Accuracy', color='orange')
        ax2.set_title('Training Accuracy Over Epochs')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()

        # Save the plot
        plot_path = os.path.join(self.save_dir, 'training_results.png')
        plt.tight_layout()
        plt.savefig(plot_path)  # Save the plot
        plt.close(fig)  # Close the figure to free memory
        print(f'Saved training results plot to {plot_path}')