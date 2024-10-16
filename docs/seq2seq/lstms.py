import torch
import torch.nn as nn

class LSTMCellManual(nn.Module):
    """
    A manual implementation of an LSTM cell.

    Args:
        input_dim (int): The dimensionality of the input feature vector.
        hidden_dim (int): The dimensionality of the hidden state.
    """
    def __init__(self, input_dim, hidden_dim):
        super(LSTMCellManual, self).__init__()
        self.input_dim = input_dim  # Set the input dimension
        self.hidden_dim = hidden_dim  # Set the hidden dimension
        
        # Linear transformations for input and hidden states for input gate, forget gate, output gate, and cell gate
        self.i2i = nn.Linear(input_dim, hidden_dim)  # Input gate
        self.h2i = nn.Linear(hidden_dim, hidden_dim)  # Hidden state to input gate
        self.i2f = nn.Linear(input_dim, hidden_dim)  # Forget gate
        self.h2f = nn.Linear(hidden_dim, hidden_dim)  # Hidden state to forget gate
        self.i2o = nn.Linear(input_dim, hidden_dim)  # Output gate
        self.h2o = nn.Linear(hidden_dim, hidden_dim)  # Hidden state to output gate
        self.i2g = nn.Linear(input_dim, hidden_dim)  # Cell gate
        self.h2g = nn.Linear(hidden_dim, hidden_dim)  # Hidden state to cell gate
        
    def forward(self, x_t, h_t, c_t):
        """
        Forward pass for the LSTM cell.

        Args:
            x_t (torch.Tensor): Input tensor of shape (batch_size, input_dim).
            h_t (torch.Tensor): Previous hidden state tensor of shape (batch_size, hidden_dim).
            c_t (torch.Tensor): Previous cell state tensor of shape (batch_size, hidden_dim).

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: The updated hidden state tensor of shape (batch_size, hidden_dim).
                - torch.Tensor: The updated cell state tensor of shape (batch_size, hidden_dim).
        """
        batch_size = x_t.size(0)  # Get the batch size from the input tensor
        
        # Assert statements to check the dimensions of inputs and hidden states
        assert x_t.size(1) == self.input_dim, f'Input dimension was expected to be {self.input_dim}, got {x_t.size(1)}'
        assert h_t.size(0) == batch_size, f'0th dimension of h_t is expected to be {batch_size}, got {h_t.size(0)}'
        assert h_t.size(1) == self.hidden_dim, f'Hidden dimension was expected to be {self.hidden_dim}, got {h_t.size(1)}'
        assert c_t.size(0) == batch_size, f'0th dimension of c_t is expected to be {batch_size}, got {c_t.size(0)}'
        assert c_t.size(1) == self.hidden_dim, f'Hidden dimension was expected to be {self.hidden_dim}, got {c_t.size(1)}'
        
        # Calculate the input gate, forget gate, output gate, and cell gate using sigmoid and tanh activations
        i_t = torch.sigmoid(self.i2i(x_t) + self.h2i(h_t))  # Input gate
        f_t = torch.sigmoid(self.i2f(x_t) + self.h2f(h_t))  # Forget gate
        g_t = torch.tanh(self.i2g(x_t) + self.h2g(h_t))     # Cell gate
        o_t = torch.sigmoid(self.i2o(x_t) + self.h2o(h_t))  # Output gate
        
        # Update the cell state
        c_t = f_t * c_t + i_t * g_t
        # Update the hidden state
        h_t = o_t * torch.tanh(c_t)
        
        return h_t, c_t  # Return the updated hidden and cell states
    
    def initialize(self, batch_size, device=None):
        """
        Initialize the hidden and cell states.

        Args:
            batch_size (int): The size of the batch.
            device (torch.device, optional): The device to allocate the tensors. Defaults to None.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: The initialized hidden state tensor filled with zeros of shape (batch_size, hidden_dim).
                - torch.Tensor: The initialized cell state tensor filled with zeros of shape (batch_size, hidden_dim).
        """
        # Return zero tensors for the initial states allocated on the specified device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))
