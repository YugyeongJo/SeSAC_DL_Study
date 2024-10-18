import os
import random 
import torch 
import torch.nn as nn 
import matplotlib.pyplot as plt

class Seq2SeqTrainer:
    def __init__(self, model, train_loader, valid_loader, optimizer, criterion, device, encoder_model_name, decoder_model_name, attention_model_name):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
        # 모델 이름 저장
        self.encoder_model_name = encoder_model_name
        self.decoder_model_name = decoder_model_name
        self.attention_model_name = attention_model_name
        
        # 저장할 폴더 경로 설정
        self.save_dir = 'saved_models'
        os.makedirs(self.save_dir, exist_ok=True)  # 폴더가 없으면 생성
        
        self.train_losses = []
        self.train_accuracies = []
        self.valid_losses = []
        self.valid_accuracies = []

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
                loss = self.criterion(outputs.reshape(-1, outputs.size(-1)), target[:, 1:].reshape(-1))  # Compute loss
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
            valid_loss, valid_accuracy = self.evaluate(self.valid_loader)
            self.valid_losses.append(valid_loss)
            self.valid_accuracies.append(valid_accuracy)
            print(f'Epoch [{epoch+1}/{num_epochs}], Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.4f}')
            
            # Save the model for this epoch
            epoch_model_path = os.path.join(self.save_dir, f'model_epoch_{epoch+1}.pth')
            torch.save(self.model.state_dict(), epoch_model_path)  # Save model state
            print(f'Saved model for epoch {epoch+1} to {epoch_model_path}')
        
        # Plot and save the loss and accuracy for train and validation after all epochs
        self.plot()

    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for source, target in data_loader:
                source, target = source.to(self.device), target.to(self.device)
                outputs = self.model(source, target[:, :-1])
                loss = self.criterion(outputs.reshape(-1, outputs.size(-1)), target[:, 1:].reshape(-1))
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs, dim=-1)
                correct_predictions += (predicted == target[:, 1:]).sum().item()
                total_predictions += target[:, 1:].numel()
        
        avg_loss = total_loss / len(data_loader)
        accuracy = correct_predictions / total_predictions
        return avg_loss, accuracy

    def plot(self):
        # Create a figure for loss and accuracy
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Plotting the training and validation loss
        ax1.plot(self.train_losses, label='Training Loss', color='blue')
        ax1.plot(self.valid_losses, label='Validation Loss', color='red')
        ax1.set_title(f'Loss Over Epochs\nEncoder: {self.encoder_model_name}, Decoder: {self.decoder_model_name}, Attention: {self.attention_model_name}')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Plotting the training and validation accuracy
        ax2.plot(self.train_accuracies, label='Training Accuracy', color='orange')
        ax2.plot(self.valid_accuracies, label='Validation Accuracy', color='green')
        ax2.set_title(f'Accuracy Over Epochs\nEncoder: {self.encoder_model_name}, Decoder: {self.decoder_model_name}, Attention: {self.attention_model_name}')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()

        # Save the plot
        plot_path = os.path.join(self.save_dir, 'final_loss_accuracy.png')
        plt.savefig(plot_path)
        plt.show()  # Display the plot
