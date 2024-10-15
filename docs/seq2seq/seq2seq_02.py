import torch
import torch.nn as nn
import random

from data_handler_02 import Vocabulary
from rnn_cells_02 import RNNCellManual, LSTMCellManual

class EncoderState:
    def __init__(self, hidden, **kargs):
        self.hidden = hidden
        self.extra_info = kargs
        for k, v in kargs.items():
            exec(f'self.{k} = v')
            
    def initialize(self):
        assert 'model_type' in dir(self)
        return self.model_type.initialize()
        
class Encoder(nn.Module):
    def __init__(self, source_vocab, embedding_dim, hidden_dim, model_type):
        super(Encoder, self).__init__()
        self.source_vocab = source_vocab
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.model_type = model_type
        
        self.embedding = nn.Embedding(source_vocab.vocab_size, embedding_dim)
        self.cell = model_type(embedding_dim, hidden_dim)
        
    def forward(self, source):
        batch_size, seq_length = source.size()
        
        embedded = self.embedding(source)
        encoder_state = self.cell.initialize(batch_size)
        
        for t in range(seq_length):
            x_t = embedded[:, t, :]      
            if self.model_type == RNNCellManual:
                encoder_state = self.cell(x_t, encoder_state)
            elif self.model_type == LSTMCellManual:
                encoder_state = self.cell(x_t, *encoder_state)
            
        return encoder_state

class Decoder(nn.Module):
    def __init__(self, target_vocab, embedding_dim, hidden_dim, model_type):
        super(Decoder, self).__init__()
        self.target_vocab = target_vocab
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self. model_type = model_type
        
        self.embedding = nn.Embedding(target_vocab.vocab_size, embedding_dim)
        self.cell = model_type(embedding_dim, hidden_dim)
        self.h2o = nn.Linear(hidden_dim, target_vocab.vocab_size)
        
    def forward(self, target, encoder_last_state, teacher_forcing_ratio=0.5):
        batch_size, seq_length = target.size()
        
        outputs = []
        
        input = torch.tensor([self.target_vocab.SOS_IDX for _ in range(batch_size)])
        decoder_state = encoder_last_state
        
        for t in range(seq_length):
            embedded = self.embedding(input)
            if self.model_type == RNNCellManual:
                decoder_state = self.cell(embedded, decoder_state)
            elif self.model_type == LSTMCellManual:
                decoder_state = self.cell(embedded, *decoder_state)
            output = self.h2o(decoder_state)
            outputs.append(output)
            
            if random.random() < teacher_forcing_ratio and t < seq_length-1:    # do teacher forcing
                input = target[:, t+1]
            else:
                input = torch.argmax(output, dim = 1)
                
        return torch.stack(outputs, dim = 1)
        
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, source, target):
        encoder_hidden = self.encoder(source)
        outputs = self.decoder(target, encoder_hidden)
        
        return outputs
        
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    from data_handler import parse_file
    from rnn_cells import RNNCellManual, LSTMCellManual
    
    EMBEDDING_DIM = 256
    BATCH_SIZE = 32
    ENCODER_MODEL = RNNCellManual
    DECODER_MODEL = RNNCellManual
    HIDDEN_DIM = 128
    criterion = nn.CrossEntropyLoss
    optimizer = torch.optim.Adam
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 10
    
    (train, valid, test), source_vocab, target_vocab = parse_file('./dataset/kor.txt', batch_size = BATCH_SIZE)
    encoder = Encoder(source_vocab, EMBEDDING_DIM, HIDDEN_DIM, ENCODER_MODEL)
    decoder = Decoder(target_vocab, EMBEDDING_DIM, HIDDEN_DIM, DECODER_MODEL)

    model = Seq2Seq(encoder = encoder, 
                    decoder = decoder)
    
    model.train()
    loss = 0
    loss_history = []
    
    criterion = criterion()
    optimizer = optimizer(model.parameters(), lr = LEARNING_RATE)
    
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_loss = 0
        for step_idx, (source_batch, target_batch) in enumerate(train):
            optimizer.zero_grad()
            
            pred_batch = model(source_batch, target_batch)
            batch_size, seq_length = target_batch.size()
            
            loss = criterion(pred_batch.view(batch_size * seq_length, -1), target_batch.view(-1))
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss
            
            if step_idx % 100 == 0:
                print(f'Epoch  {epoch} / {NUM_EPOCHS + 1}, step {step_idx}: loss - {loss}')
                
        avg_loss = epoch_loss.item() / len(train)
        loss_history.append(avg_loss)

    plt.plot(loss_history)
    plt.show()