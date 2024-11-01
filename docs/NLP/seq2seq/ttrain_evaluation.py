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
        
        # colab 버전
        # self.save_dir = '/content/drive/My Drive/my_model'
        # os.makedirs(self.save_dir, exist_ok=True)

        
        self.train_losses = []
        self.train_accuracies = []
        self.train_bleu_scores = []
        self.valid_losses = []
        self.valid_accuracies = []
        self.valid_bleu_scores = []
        
    def train(self, num_epochs):
        
        for epoch in range(num_epochs):
            self.model.train()
            
            train_loss, train_prediction, train_target = self.train_or_evaluate(self.train_loader, update = True, evaluate = True)
            
            # print("=====")
            # print(f"train_prediction : {train_prediction}")
            # assert len(train_prediction) == 0
            # print(f"train_target : {train_target}")
            # print("=====")
            
            train_accuracy = self.calculate_accuracy(train_prediction, train_target)
            
            train_blue_score = self.calculate_bleu(train_prediction, train_target)
            
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_accuracy)
            self.train_bleu_scores.append(train_blue_score)
            
            with open(os.path.join(self.save_dir, f'Train_translations {epoch+1}.txt'), 'w+', encoding = 'utf-8') as f:
                f.write(f'Epoch {epoch + 1} Translations:\n')
                
                for i in range(len(train_prediction)):
                    f.write(f'Predicted: {train_prediction[i]}\n')
                    f.write(f'Target: {train_target[i]}\n\n')
                    f.write("\n")
                    
            with torch.no_grad():
                self.model.eval()
                valid_loss, valid_prediction, valid_target = self.train_or_evaluate(self.valid_loader, update = False, evaluate = True)
                
                valid_accuracy = self.calculate_accuracy(valid_prediction, valid_target)
            
                valid_blue_score = self.calculate_bleu(valid_prediction, valid_target)
                
                self.valid_losses.append(valid_loss)
                self.valid_accuracies.append(valid_accuracy)
                self.valid_bleu_scores.append(valid_blue_score)
                
            with open(os.path.join(self.save_dir, f'Valid_translations {epoch+1}.txt'), 'w+', encoding = 'utf-8') as f:
                f.write(f'Epoch {epoch + 1} Translations:\n')
                
                for i in range(len(valid_prediction)):
                    f.write(f'Predicted: {valid_prediction[i]}\n')
                    f.write(f'Target: {valid_target[i]}\n\n')
                    f.write("\n")
                    
            # Print training and validation metrics for the current epoch
            print(f'Epoch {epoch + 1}/{num_epochs}')
            print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train BLEU: {train_blue_score:.4f}')
            print(f'Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.4f}, Valid BLEU: {valid_blue_score:.4f}')
                    
            # 모델 저장 추가
            model_save_path = os.path.join(self.save_dir, f'model_epoch_{epoch + 1}.pth')
            torch.save(self.model.state_dict(), model_save_path)
            print(f'Model saved to {model_save_path}')
            
        # Plot and save the loss and accuracy for train and validation after all epochs
        self.plot()
                
    def train_or_evaluate(self, data, update = False, evaluate = False):
        predicted_result = []
        target_result = []
        epoch_loss = 0
        
        # from itertools import islice
        # for idx, (source, target) in enumerate(islice(data, 5)):
        for idx, (source, target) in enumerate(data):
            if update:
                self.optimizer.zero_grad()
                
            source, target = source.to(self.device), target.to(self.device)
            outputs = self.model(source, target)
            loss = self.criterion(outputs.reshape(-1, outputs.size(-1)), target.reshape(-1))
            
            if update:
                loss.backward()
                self.optimizer.step()
                
            epoch_loss += loss.item()
            
            # print(f'loss for step {idx} : {loss.item()}')
            
            if evaluate:
                _, predicted = torch.max(outputs, dim=-1)
                
                # print("=====")
                # print(f"predicted : ", predicted)
                
                predicted_sentences = [[self.target_vocab.idx2word[idx] for idx in sent] for sent in predicted.tolist()]
                target_sentences = [[self.target_vocab.idx2word[idx] for idx in sent] for sent in target.tolist()]
                
                predicted_result.extend(predicted_sentences)
                target_result.extend(target_sentences)

        return epoch_loss, predicted_result, target_result
        
    def calculate_accuracy(self, predicted_sentences, target_sentences):
        """Calculate the accuracy given the predicted and target sentences, excluding PAD tokens."""
        correct_word_count = 0
        total_word_count = 0

        # Define the special tokens to exclude
        special_tokens = [self.target_vocab.idx2word[self.target_vocab.PAD_IDX], 
                        self.target_vocab.idx2word[self.target_vocab.EOS_IDX], 
                        self.target_vocab.idx2word[self.target_vocab.SOS_IDX]]

        # Iterate through each predicted and target sentence pair (1st for loop)
        for pred, target in zip(predicted_sentences, target_sentences):
            # Remove special tokens from both predicted and target sentences
            pred_filtered = [word for word in pred if word not in special_tokens]
            target_filtered = [word for word in target if word not in special_tokens]
        
            # Compare word by word between filtered predicted and target sentences
            for pred_word, target_word in zip(pred_filtered, target_filtered):
                if pred_word == target_word:
                    correct_word_count += 1

            # Total word count should be the number of words in the target (excluding special tokens)
            total_word_count += len(target_filtered)

        # Calculate accuracy as the ratio of correct words to total words
        accuracy = correct_word_count / total_word_count if total_word_count > 0 else 0
        return accuracy

    def calculate_bleu(self, predicted_sentences, target_sentences):
        """Calculate BLEU score given predicted sentences and target sentences, excluding PAD tokens."""
        total_bleu = 0
        total_samples = len(predicted_sentences)

        for pred, target in zip(predicted_sentences, target_sentences):
            # Remove PAD tokens from both the predicted and target sentence
            special_tokens = [self.target_vocab.idx2word[self.target_vocab.PAD_IDX], 
                              self.target_vocab.idx2word[self.target_vocab.EOS_IDX], 
                              self.target_vocab.idx2word[self.target_vocab.SOS_IDX], ]
            pred_filtered = [word for word in pred if word not in special_tokens]
            target_filtered = [word for word in target if word not in special_tokens]

            # Calculate BLEU score with the filtered sentences
            total_bleu += bleu(pred_filtered, target_filtered)

        # Calculate the average BLEU score across all samples
        return total_bleu / total_samples if total_samples > 0 else 0
    
    def plot(self):
        # Create a figure for loss and accuracy
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

        # Set the main title for the figure
        fig.suptitle(f'Training and Validation Metrics\nEncoder: {self.encoder_model_name}, Decoder: {self.decoder_model_name}, Attention: {self.attention_model_name}', fontsize=16)
        
        # Plotting the training and validation loss
        ax1.plot(self.train_losses, label='Training Loss', color='blue')
        ax1.plot(self.valid_losses, label='Validation Loss', color='orange')
        ax1.set_title('Loss Over Epochs')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Plotting the training and validation accuracy        
        ax2.plot(self.train_accuracies, label='Training Accuracy', color='blue')
        ax2.plot(self.valid_accuracies, label='Validation Accuracy', color='orange')
        ax2.set_title('Accuracy Over Epochs')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()

        # Plotting BLEU scores
        ax3.plot(self.train_bleu_scores, label='Training BLEU Score', color='blue')
        ax3.plot(self.valid_bleu_scores, label='Validation BLEU Score', color='orange')
        ax3.set_title('BLEU Score Over Epochs')
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('BLEU Score')
        ax3.legend()
        
        # Save the plot
        plot_path = os.path.join(self.save_dir, 'final_plot.png')
        plt.savefig(plot_path)
        plt.show()  # Display the plot