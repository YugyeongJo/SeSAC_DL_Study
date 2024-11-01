import torch
import torch.nn as nn

import torch
import torch.nn as nn

class LuongAttention(nn.Module):
    """
    Luong Attention Mechanism.

    This attention mechanism computes the context vector based on the current decoder state
    and the encoder hidden states.

    Args:
        None
    """
    def __init__(self):
        super(LuongAttention, self).__init__()
        
    def forward(self, decoder_state, encoder_hiddens):
        """
        Forward pass for the Luong attention mechanism.

        Args:
            decoder_state (torch.Tensor): The current decoder hidden state of shape (batch_size, hidden_dim).
            encoder_hiddens (torch.Tensor): The encoder hidden states of shape (batch_size, encoder_sequence_length, hidden_dim).

        Returns:
            torch.Tensor: The computed context vector of shape (batch_size, hidden_dim).
        """
        batch_size, encoder_sequence_length, encoder_hidden_dim = encoder_hiddens.size()
        
        # Initialize attention scores
        attention_score = torch.zeros(batch_size, encoder_sequence_length, device=decoder_state.device)
        s_t = decoder_state  # Current decoder state
        
        # Calculate attention scores
        for t in range(encoder_sequence_length):
            h_t = encoder_hiddens[:, t]  # Get the hidden state at time t
            attention_score[:, t] = torch.sum(s_t * h_t, dim=1)  # Element-wise product and sum
        
        # Compute attention distribution using softmax
        attention_distribution = torch.softmax(attention_score, dim=1)
        
        # Initialize context vector
        context_vector = torch.zeros(batch_size, encoder_hidden_dim, device=decoder_state.device)
        
        # Compute the context vector as a weighted sum of encoder hidden states
        for t in range(encoder_sequence_length):
            context_vector += attention_distribution[:, t].unsqueeze(1) * encoder_hiddens[:, t]
            
        return context_vector  # Return the context vector


class BahdanauAttention(nn.Module):
    """
    Bahdanau Attention Mechanism.

    This attention mechanism computes the context vector using learned weights and the current decoder state.

    Args:
        k (int): The hidden dimension for attention.
        h (int): The hidden dimension of the encoder's output.
    """
    def __init__(self, k, h):
        super(BahdanauAttention, self).__init__()
        self.W_a = nn.Linear(k, 1)  # Linear layer for attention score
        self.W_b = nn.Linear(h, k)   # Linear layer for decoder state transformation
        self.W_c = nn.Linear(h, k)   # Linear layer for encoder hidden states transformation
        
    def forward(self, decoder_state, encoder_hiddens):
        """
        Forward pass for the Bahdanau attention mechanism.

        Args:
            decoder_state (torch.Tensor): The current decoder hidden state of shape (batch_size, hidden_dim).
            encoder_hiddens (torch.Tensor): The encoder hidden states of shape (batch_size, encoder_sequence_length, hidden_dim).

        Returns:
            torch.Tensor: The computed context vector of shape (batch_size, hidden_dim).
        """
        batch_size, encoder_sequence_length, encoder_hidden_dim = encoder_hiddens.size()
        
        # Compute the attention scores using learned weights
        attention_score = self.W_a(torch.tanh(self.W_b(decoder_state).unsqueeze(1) + self.W_c(encoder_hiddens))).squeeze(2)
        attention_distribution = torch.softmax(attention_score, dim=1)  # Normalize to get probabilities
        
        # Initialize context vector
        context_vector = torch.zeros(batch_size, encoder_hidden_dim, device=decoder_state.device)
        
        # Compute the context vector as a weighted sum of encoder hidden states
        for t in range(encoder_sequence_length):
            context_vector += attention_distribution[:, t].unsqueeze(1) * encoder_hiddens[:, t]
            
        return context_vector  # Return the context vector


# # for문 대신 torch.bmm 사용 code
# class LuongAttention(nn.Module):
#     """
#     Luong Attention Mechanism.

#     This attention mechanism computes the context vector based on the current decoder state
#     and the encoder hidden states.

#     Args:
#         None
#     """
#     def __init__(self):
#         super(LuongAttention, self).__init__()
        
#     def forward(self, decoder_state, encoder_hiddens):
#         """
#         Forward pass for the Luong attention mechanism.

#         Args:
#             decoder_state (torch.Tensor): The current decoder hidden state of shape (batch_size, hidden_dim).
#             encoder_hiddens (torch.Tensor): The encoder hidden states of shape (batch_size, encoder_sequence_length, hidden_dim).

#         Returns:
#             torch.Tensor: The computed context vector of shape (batch_size, hidden_dim).
#         """
#         # Get the dimensions
#         batch_size, encoder_sequence_length, encoder_hidden_dim = encoder_hiddens.size()
        
#         # Compute attention scores using batch matrix multiplication
#         attention_score = torch.bmm(encoder_hiddens, decoder_state.unsqueeze(2)).squeeze(2)
        
#         # Compute attention distribution using softmax
#         attention_distribution = torch.softmax(attention_score, dim=1)
        
#         # Compute the context vector as a weighted sum of encoder hidden states
#         context_vector = torch.bmm(attention_distribution.unsqueeze(1), encoder_hiddens).squeeze(1)
        
#         return context_vector  # Return the context vector

# class BahdanauAttention(nn.Module):
#     """
#     Bahdanau Attention Mechanism.

#     This attention mechanism computes the context vector using learned weights and the current decoder state.

#     Args:
#         k (int): The hidden dimension for attention.
#         h (int): The hidden dimension of the encoder's output.
#     """
#     def __init__(self, k, h):
#         super(BahdanauAttention, self).__init__()
#         self.W_a = nn.Linear(k, 1)  # Linear layer for attention score
#         self.W_b = nn.Linear(h, k)   # Linear layer for decoder state transformation
#         self.W_c = nn.Linear(h, k)   # Linear layer for encoder hidden states transformation
        
#     def forward(self, decoder_state, encoder_hiddens):
#         """
#         Forward pass for the Bahdanau attention mechanism.

#         Args:
#             decoder_state (torch.Tensor): The current decoder hidden state of shape (batch_size, hidden_dim).
#             encoder_hiddens (torch.Tensor): The encoder hidden states of shape (batch_size, encoder_sequence_length, hidden_dim).

#         Returns:
#             torch.Tensor: The computed context vector of shape (batch_size, hidden_dim).
#         """
#         # Get the dimensions
#         batch_size, encoder_sequence_length, encoder_hidden_dim = encoder_hiddens.size()
        
#         # Repeat decoder state across the sequence length for element-wise addition
#         decoder_state_expanded = decoder_state.unsqueeze(1).expand(-1, encoder_sequence_length, -1)
        
#         # Compute the attention scores using learned weights
#         energy = torch.tanh(self.W_b(decoder_state_expanded) + self.W_c(encoder_hiddens))
#         attention_score = self.W_a(energy).squeeze(2)
        
#         # Normalize the attention scores to get probabilities
#         attention_distribution = torch.softmax(attention_score, dim=1)
        
#         # Compute the context vector as a weighted sum of encoder hidden states
#         context_vector = torch.bmm(attention_distribution.unsqueeze(1), encoder_hiddens).squeeze(1)
        
#         return context_vector  # Return the context vector
