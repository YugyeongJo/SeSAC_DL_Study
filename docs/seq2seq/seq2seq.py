import torch
import torch.nn as nn

class EncoderState:
    def __init__(self, hidden, **kargs):
        self.hidden = hidden 
        for k, v in kargs.items():
            exec(f'self.{k} = v')
            
    def initialize(self):
        assert 'model_type' in dir(self)
        return self.model_type.initialize()
        
class Encoder(nn.Module):
    def __init__(self, source_vocab, embedding_dim, hidden_dim, model_type, ):
        super(Encoder, self).__init__()
        self.source_vocab = source_vocab
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(source_vocab.vocab_size, embedding_dim)
        self.cell = model_type(embedding_dim, hidden_dim)
        
    def forward(self, source):
        batch_size, seq_length = source.size()
        
        embedded = self.embedding(source)
        encoder_state = EncoderState(model_type = self.model_type).initialize()
        
        for t in range(seq_length):
            x_t = embedded[:, t, :]                
            encoder_state = self.cell(x_t, encoder_state)
            
        return encoder_state

class Decoder(nn.Module):
    def __init__(self, ):
        super(Decoder, self).__init__()
        
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, source, target):
        encoder_hidden = self.encoder(source)
        
if __name__ == '__main__':
    encoder = Encorder(source_vocab, embedding_dim, hidden_dim, RNNCellManual)
    model = Seq2Seq