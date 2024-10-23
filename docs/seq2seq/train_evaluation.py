import os
import torch 
import torch.nn as nn 
import matplotlib.pyplot as plt

from data_handler import Vocabulary
from metrics import bleu

class Seq2SeqTrainer:
    def __init__(self, model, train_loader, valid_loader, optimizer, criterion, device, encoder_model_name, decoder_model_name, attention_model_name, source_vocab, target_vocab):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
        # Vocabulary PAD_IDX
        self.PAD_IDX = Vocabulary.PAD_IDX
        
        # 모델 이름 저장
        self.encoder_model_name = encoder_model_name
        self.decoder_model_name = decoder_model_name
        self.attention_model_name = attention_model_name
        
        # 단어사전 저장
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        
        # 저장할 폴더 경로 설정
        self.save_dir = 'saved_models'
        os.makedirs(self.save_dir, exist_ok=True)  # 폴더가 없으면 생성
        
        self.train_losses = []
        self.train_accuracies = []
        self.valid_losses = []
        self.valid_accuracies = []
        self.bleu_scores = []

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
                
                # Calculate accuracy while ignoring PAD_IDX
                _, predicted = torch.max(outputs, dim=-1)  # Get predicted classes
                mask = (target[:, 1:] != self.PAD_IDX)  # Create a mask to ignore PAD_IDX positions
                correct_predictions += ((predicted == target[:, 1:]) & mask).sum().item()
                total_predictions += mask.sum().item()  # Only count non-PAD_IDX positions
            
            # Calculate average loss and accuracy for the epoch
            avg_loss = epoch_loss / len(self.train_loader)
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            self.train_losses.append(avg_loss)
            self.train_accuracies.append(accuracy)
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Train Accuracy: {accuracy:.4f}')
            
            # Get predictions for evaluation
            predicted_sentences = []
            target_sentences = []

            with torch.no_grad():
                for source, target in self.valid_loader:
                    source, target = source.to(self.device), target.to(self.device)
                    outputs = self.model(source, target[:, :-1])
                    _, predicted = torch.max(outputs, dim=-1)
                    print(predicted.dtype)
                    
                    predicted_sentences.append(predicted)
                    target_sentences.append(target[:, 1:])  # Collect target sentences
                
            # Evaluate on validation set
            valid_loss, valid_accuracy = self.evaluate(predicted_sentences, target_sentences)
            self.valid_losses.append(valid_loss)
            self.valid_accuracies.append(valid_accuracy)
            print(f'Epoch [{epoch+1}/{num_epochs}], Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.4f}')
            
            # Calculate BLEU score for the validation set
            bleu_score = self.calculate_bleu(predicted_sentences, target_sentences)
            self.bleu_scores.append(bleu_score)
            print(f'Epoch [{epoch+1}/{num_epochs}], BLEU Score: {bleu_score:.4f}')
            
            # Save the model for this epoch
            epoch_model_path = os.path.join(self.save_dir, f'model_epoch_{epoch+1}.pth')
            torch.save(self.model.state_dict(), epoch_model_path)  # Save model state
            print(f'Saved model for epoch {epoch+1} to {epoch_model_path}')

            # Save predictions and targets to a text file
            self.save_translations(epoch, predicted_sentences, target_sentences)
            
        # Plot and save the loss and accuracy for train and validation after all epochs
        self.plot()

    def evaluate(self, predicted_sentences, target_sentences):
        """Evaluate the model using predictions and targets directly."""
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        for i in range(len(predicted_sentences)):
            predicted = predicted_sentences[i]
            target = target_sentences[i]

            # Calculate loss (assuming 'outputs' was computed similarly)
            loss = self.criterion(predicted.reshape(-1, predicted.size(-1)), target.reshape(-1))
            total_loss += loss.item()

            mask = (target != self.PAD_IDX)  # Create a mask to ignore PAD_IDX positions
            correct_predictions += ((predicted == target) & mask).sum().item()  # Count correct predictions while ignoring PAD_IDX
            total_predictions += mask.sum().item()  # Only count non-PAD_IDX positions
        
        avg_loss = total_loss / len(predicted_sentences)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        return avg_loss, accuracy
    
    def evaluate_or_train(self, data, update = False, evaluate = False):
        res = []
        answer = []
        loss = 0
        for source, target in data:
            if update:
                self.optimizer.zero_grad()
            source, target = source.to(self.device), target.to(self.device)
            outputs = self.model(source, target[:,:-1])
            loss = self.criterion(outputs.reshape(-1, outputs.size(-1)), target[:, 1:].reshape(-1))  # Compute loss
            
            if update:
                loss.backward()
                self.optimizer.step() 
            loss += loss.item()
                
            if evaluate:
                _, predicted = torch.max(outputs, dim=-1)  # Get predicted classes
                predicted_sentences = [[self.target_vocab.idx2word[idx] for idx in sent] for sent in predicted.tolist()]
                target_sentences = [[self.target_vocab.idx2word[idx] for idx in sent] for sent in target.tolist()]
                res.extend(predicted_sentences)
                answer.extend(target_sentences)
                return loss, res, answer 
            return loss

    def calculate_bleu(self, predicted_sentences, target_sentences):
        """Calculate BLEU score given predicted sentences and target sentences."""
        total_bleu = 0
        total_samples = len(predicted_sentences)

        for i in range(total_samples):
            total_bleu += bleu(predicted_sentences[i], target_sentences[i])  # BLEU 점수 계산

        return total_bleu / total_samples if total_samples > 0 else 0

    def translator(self, indices, target_vocab):
        """Convert a list of indices back to a sentence string using the target vocabulary."""
        return ' '.join(target_vocab[idx] for idx in indices if idx != self.PAD_IDX)

    def save_translations(self, epoch, predicted_sentences, target_sentences):
        """Save predictions and targets to a text file after each epoch."""
        with torch.no_grad():
            with open(os.path.join(self.save_dir, 'translations.txt'), 'a') as f:
                f.write(f'Epoch {epoch + 1} Translations:\n')

                for i in range(len(predicted_sentences)):
                    predicted_sentence = self.translator(predicted_sentences[i].cpu().numpy(), self.target_vocab)
                    target_sentence = self.translator(target_sentences[i].cpu().numpy(), self.target_vocab)  # Convert target sentence

                    f.write(f'Predicted: {predicted_sentence}\n')
                    f.write(f'Target: {target_sentence}\n\n')
                        
    def plot(self):
        # Create a figure for loss and accuracy
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

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

        # Plotting BLEU scores
        ax3.plot(self.bleu_scores, label='BLEU Score', color='purple')
        ax3.set_title('BLEU Score Over Epochs')
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('BLEU Score')
        ax3.legend()
        
        # Save the plot
        plot_path = os.path.join(self.save_dir, 'final_loss_accuracy.png')
        plt.savefig(plot_path)
        plt.show()  # Display the plot
