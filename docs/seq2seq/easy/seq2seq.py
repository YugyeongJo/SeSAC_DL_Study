import random 
import torch 
import torch.nn as nn 

from rnns import RNNCellManual
from lstms import LSTMCellManual
from attentions import LuongAttention, BahdanauAttention

class EncoderState:
    """
    Represents the state of the encoder.

    Attributes can be initialized dynamically using keyword arguments.
    """
    def __init__(self, **kargs):
        for k, v in kargs.items():
            exec(f'self.{k} = v')
            
    def initialize(self):
        """
        Initialize the encoder state based on the specified model type.
        
        Returns:
            The initialized state of the encoder.
        """
        assert 'model_type' in dir(self)
        return self.model_type.initialize()
    
class Encoder(nn.Module):
    """
    Encoder module for the Seq2Seq model.

    Args:
        source_vocab (Vocabulary): The source vocabulary for the input data.
        embedding_dim (int): The dimensionality of the embeddings.
        hidden_dim (int): The dimensionality of the hidden states.
        model_type (nn.Module): The RNN cell type (e.g., RNNCellManual, LSTMCellManual).
    """
    def __init__(self, source_vocab, embedding_dim, hidden_dim, model_type, device):
        super(Encoder, self).__init__()
        self.source_vocab = source_vocab
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.model_type = model_type
        self.device = device
        
        # Embedding layer for source input
        self.embedding = nn.Embedding(source_vocab.vocab_size, embedding_dim).to(self.device)
        # RNN cell (RNN or LSTM) for the encoder
        self.cell = model_type(embedding_dim, hidden_dim).to(self.device)
        
    def forward(self, source):
        """
        Forward pass for the encoder.

        Args:
            source (torch.Tensor): Input source tensor of shape (batch_size, seq_length).

        Returns:
            torch.Tensor: Hidden states of the encoder of shape (batch_size, seq_length, hidden_dim).
        """
        batch_size, seq_length = source.size()
        hiddens = []  # List to store hidden states
        
        embedded = self.embedding(source).to(self.device)  # Get embedded representation
        encoder_state = self.cell.initialize(batch_size).to(self.device)  # Initialize the cell state
        
        # Iterate through each time step in the input sequence
        for t in range(seq_length):
            x_t = embedded[:, t, :]  # Get the input at time step t
            if self.model_type == RNNCellManual:
                encoder_state = self.cell(x_t, encoder_state)  # Update the RNN state
                hiddens.append(encoder_state)  # Store the hidden state
            elif self.model_type == LSTMCellManual:
                encoder_state = self.cell(x_t, encoder_state)  # Update the LSTM state
                hiddens.append(encoder_state[0])  # Store the hidden state (cell state is not needed)
                
        return torch.stack(hiddens, dim=1).to(self.device)  # Return stacked hidden states
    
class Decoder(nn.Module):
    """
    Decoder module for the Seq2Seq model with attention mechanism.

    Args:
        target_vocab (Vocabulary): The target vocabulary for the output data.
        embedding_dim (int): The dimensionality of the embeddings.
        hidden_dim (int): The dimensionality of the hidden states.
        model_type (nn.Module): The RNN cell type (e.g., RNNCellManual, LSTMCellManual).
        attention (nn.Module): The attention mechanism to be used (LuongAttention or BahdanauAttention).
    """
    def __init__(self, target_vocab, embedding_dim, hidden_dim, model_type, attention, device):
        super(Decoder, self).__init__()
        
        self.target_vocab = target_vocab 
        self.embedding_dim = embedding_dim 
        self.hidden_dim = hidden_dim 
        self.model_type = model_type    
        self.device = device
        
        # Initialize attention mechanism and related layers
        if attention == LuongAttention:
            self.attention = LuongAttention()
            self.W_c = nn.Linear(target_vocab.vocab_size + hidden_dim, target_vocab.vocab_size).to(self.device)
            self.embedding = nn.Embedding(target_vocab.vocab_size, embedding_dim).to(self.device)
            self.cell = model_type(embedding_dim, hidden_dim).to(self.device)
        elif attention == BahdanauAttention:
            self.attention = BahdanauAttention()
            self.cell = model_type(embedding_dim + hidden_dim, hidden_dim).to(self.device)
        else:
            self.embedding = nn.Embedding(target_vocab.vocab_size, embedding_dim).to(self.device)
            self.cell = model_type(embedding_dim, hidden_dim).to(self.device)
            
        self.h2o = nn.Linear(hidden_dim, target_vocab.vocab_size).to(self.device)  # Linear layer for output predictions
            
    def forward(self, target, encoder_hiddens, teacher_forcing_ratio=0.5):
        """
        Forward pass for the decoder.

        Args:
            target (torch.Tensor): Target tensor of shape (batch_size, seq_length).
            encoder_hiddens (torch.Tensor): Hidden states from the encoder of shape (batch_size, seq_length, hidden_dim).
            teacher_forcing_ratio (float): The probability of using teacher forcing during training.

        Returns:
            torch.Tensor: Output predictions from the decoder of shape (batch_size, seq_length, target_vocab_size).
        """
        encoder_last_state = encoder_hiddens[:, -1]  # Get the last encoder hidden state
        
        batch_size, seq_length = target.size()
        
        outputs = []  # List to store decoder outputs
        
        # Initialize input to the decoder with the start of sequence token
        input = torch.tensor([self.target_vocab.SOS_IDX for _ in range(batch_size)], device=self.device)
        decoder_state = encoder_last_state  # Initialize decoder state with the last encoder state
        
        # Iterate through each time step in the target sequence
        for t in range(seq_length):
            embedded = self.embedding(input).to(self.device)  # Get embedded representation
            if isinstance(self.attention, BahdanauAttention):
                context_vector = self.attention(encoder_hiddens, decoder_state).to(self.device)  # Compute context vector
                embedded = torch.cat((embedded, context_vector), dim=1)  # Concatenate embedding and context vector
                
            # Update the decoder state using the RNN cell
            if self.model_type == RNNCellManual:
                decoder_state = self.cell(embedded, decoder_state)
            elif self.model_type == LSTMCellManual:
                decoder_state = self.cell(embedded, *decoder_state)  # Update state for LSTM
            
            output = self.h2o(decoder_state)  # Get output predictions
            
            if isinstance(self.attention, LuongAttention):
                context_vector = self.attention(decoder_state, encoder_hiddens).to(self.device)  # Compute context vector
                output = torch.cat((output, context_vector), dim=1)  # Concatenate output and context vector
                output = torch.tanh(self.W_c(output))  # Apply linear transformation
            
            outputs.append(output)  # Store output predictions
            
            # Apply teacher forcing or use model predictions as the next input
            if random.random() < teacher_forcing_ratio and t < seq_length - 1:
                input = target[:, t + 1]  # Use the target for the next time step
            else: 
                input = torch.argmax(output, dim=1)  # Use model predictions
            
        return torch.stack(outputs, dim=1).to(self.device)  # Return stacked outputs
    
class Seq2Seq(nn.Module):
    """
    Sequence-to-sequence model combining encoder and decoder.

    Args:
        encoder (Encoder): The encoder module.
        decoder (Decoder): The decoder module.
    """
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, source, target):
        """
        Forward pass for the Seq2Seq model.

        Args:
            source (torch.Tensor): Input source tensor of shape (batch_size, source_seq_length).
            target (torch.Tensor): Target tensor of shape (batch_size, target_seq_length).

        Returns:
            torch.Tensor: Output predictions from the decoder of shape (batch_size, target_seq_length, target_vocab_size).
        """
        encoder_hiddens = self.encoder(source)  # Get encoder hidden states
        outputs = self.decoder(target, encoder_hiddens)  # Get decoder outputs
        
        return outputs.to(self.encoder.device)  # Return output predictions

    
# # another code
# import random 
# import torch 
# import torch.nn as nn 

# from rnns import RNNCellManual, LSTMCellManual
# from attentions import LuongAttention, BahdanauAttention

# class EncoderState:
#     """
#     Represents the state of the encoder.

#     Attributes can be initialized dynamically using keyword arguments.
#     """
#     def __init__(self, **kargs):
#         for k, v in kargs.items():
#             setattr(self, k, v)
            
#     def initialize(self):
#         """
#         Initialize the encoder state based on the specified model type.
        
#         Returns:
#             The initialized state of the encoder.
#         """
#         assert hasattr(self, 'model_type'), "model_type attribute is required"
#         return self.model_type.initialize()
    
# class Encoder(nn.Module):
#     """
#     Encoder module for the Seq2Seq model.

#     Args:
#         source_vocab (Vocabulary): The source vocabulary for the input data.
#         embedding_dim (int): The dimensionality of the embeddings.
#         hidden_dim (int): The dimensionality of the hidden states.
#         model_type (nn.Module): The RNN cell type (e.g., RNNCellManual, LSTMCellManual).
#     """
#     def __init__(self, source_vocab, embedding_dim, hidden_dim, model_type):
#         super(Encoder, self).__init__()
#         self.source_vocab = source_vocab
#         self.embedding_dim = embedding_dim
#         self.hidden_dim = hidden_dim
#         self.model_type = model_type
        
#         # Embedding layer for source input
#         self.embedding = nn.Embedding(source_vocab.vocab_size, embedding_dim)
#         # RNN cell (RNN or LSTM) for the encoder
#         self.cell = model_type(embedding_dim, hidden_dim)
        
#     def forward(self, source):
#         """
#         Forward pass for the encoder.

#         Args:
#             source (torch.Tensor): Input source tensor of shape (batch_size, seq_length).

#         Returns:
#             torch.Tensor: Hidden states of the encoder of shape (batch_size, seq_length, hidden_dim).
#         """
#         batch_size, seq_length = source.size()
#         hiddens = []  # List to store hidden states
        
#         embedded = self.embedding(source)  # Get embedded representation
#         encoder_state = self.cell.initialize(batch_size)  # Initialize the cell state
        
#         # Iterate through each time step in the input sequence
#         for t in range(seq_length):
#             x_t = embedded[:, t, :]  # Get the input at time step t
#             encoder_state = self.cell(x_t, encoder_state)  # Update the state (RNN or LSTM)
#             hiddens.append(encoder_state[0] if isinstance(encoder_state, tuple) else encoder_state)  # Store the hidden state

#         return torch.stack(hiddens, dim=1)  # Return stacked hidden states

# class Decoder(nn.Module):
#     """
#     Decoder module for the Seq2Seq model with attention mechanism.

#     Args:
#         target_vocab (Vocabulary): The target vocabulary for the output data.
#         embedding_dim (int): The dimensionality of the embeddings.
#         hidden_dim (int): The dimensionality of the hidden states.
#         model_type (nn.Module): The RNN cell type (e.g., RNNCellManual, LSTMCellManual).
#         attention (nn.Module): The attention mechanism to be used (LuongAttention or BahdanauAttention).
#     """
#     def __init__(self, target_vocab, embedding_dim, hidden_dim, model_type, attention):
#         super(Decoder, self).__init__()
        
#         self.target_vocab = target_vocab 
#         self.embedding_dim = embedding_dim 
#         self.hidden_dim = hidden_dim 
#         self.model_type = model_type    
        
#         # Initialize attention mechanism and related layers
#         self.embedding = nn.Embedding(target_vocab.vocab_size, embedding_dim)
#         self.cell = model_type(embedding_dim + (hidden_dim if isinstance(attention, BahdanauAttention) else 0), hidden_dim)
#         self.attention = attention

#         self.h2o = nn.Linear(hidden_dim, target_vocab.vocab_size)  # Linear layer for output predictions
#         if isinstance(attention, LuongAttention):
#             self.W_c = nn.Linear(hidden_dim * 2, target_vocab.vocab_size)

#     def forward(self, target, encoder_hiddens, teacher_forcing_ratio=0.5):
#         """
#         Forward pass for the decoder.

#         Args:
#             target (torch.Tensor): Target tensor of shape (batch_size, seq_length).
#             encoder_hiddens (torch.Tensor): Hidden states from the encoder of shape (batch_size, seq_length, hidden_dim).
#             teacher_forcing_ratio (float): The probability of using teacher forcing during training.

#         Returns:
#             torch.Tensor: Output predictions from the decoder of shape (batch_size, seq_length, target_vocab_size).
#         """
#         batch_size, seq_length = target.size()
#         outputs = []  # List to store decoder outputs
        
#         # Initialize input to the decoder with the start of sequence token
#         input = torch.tensor([self.target_vocab.SOS_IDX] * batch_size).to(target.device)
#         decoder_state = encoder_hiddens[:, -1]  # Initialize decoder state with the last encoder state
        
#         # Iterate through each time step in the target sequence
#         for t in range(seq_length):
#             embedded = self.embedding(input)  # Get embedded representation
#             if isinstance(self.attention, BahdanauAttention):
#                 context_vector = self.attention(encoder_hiddens, decoder_state)  # Compute context vector
#                 embedded = torch.cat((embedded, context_vector), dim=1)  # Concatenate embedding and context vector

#             # Update the decoder state using the RNN cell
#             decoder_state = self.cell(embedded, decoder_state)
            
#             # Generate output predictions
#             output = self.h2o(decoder_state[0] if isinstance(decoder_state, tuple) else decoder_state)

#             if isinstance(self.attention, LuongAttention):
#                 context_vector = self.attention(decoder_state, encoder_hiddens)  # Compute context vector
#                 output = torch.cat((output, context_vector), dim=1)  # Concatenate output and context vector
#                 output = torch.tanh(self.W_c(output))  # Apply linear transformation
            
#             outputs.append(output)  # Store output predictions
            
#             # Apply teacher forcing or use model predictions as the next input
#             input = target[:, t] if random.random() < teacher_forcing_ratio else torch.argmax(output, dim=1)
            
#         return torch.stack(outputs, dim=1)  # Return stacked outputs

# class Seq2Seq(nn.Module):
#     """
#     Sequence-to-sequence model combining encoder and decoder.

#     Args:
#         encoder (Encoder): The encoder module.
#         decoder (Decoder): The decoder module.
#     """
#     def __init__(self, encoder, decoder):
#         super(Seq2Seq, self).__init__()
#         self.encoder = encoder
#         self.decoder = decoder
    
#     def forward(self, source, target):
#         """
#         Forward pass for the Seq2Seq model.

#         Args:
#             source (torch.Tensor): Input source tensor of shape (batch_size, source_seq_length).
#             target (torch.Tensor): Target tensor of shape (batch_size, target_seq_length).

#         Returns:
#             torch.Tensor: Output predictions from the decoder of shape (batch_size, target_seq_length, target_vocab_size).
#         """
#         encoder_hiddens = self.encoder(source)  # Get encoder hidden states
#         outputs = self.decoder(target, encoder_hiddens)  # Get decoder outputs
        
#         return outputs  # Return output predictions
