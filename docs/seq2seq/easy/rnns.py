import torch
import torch.nn as nn

class RNNCellManual(nn.Module):
    """
    A manual implementation of a simple RNN cell.

    Args:
        input_dim (int): The dimensionality of the input feature vector.
        hidden_dim (int): The dimensionality of the hidden state.
    """
    def __init__(self, input_dim, hidden_dim):
        super(RNNCellManual, self).__init__()
        self.input_dim = input_dim  # Set the input dimension
        self.hidden_dim = hidden_dim  # Set the hidden dimension
        
        # Linear transformation from input to hidden state
        self.i2h = nn.Linear(input_dim, hidden_dim)
        # Linear transformation from hidden state to hidden state
        self.h2h = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x_t, h_t):
        """
        Forward pass for the RNN cell.

        Args:
            x_t (torch.Tensor): Input tensor of shape (batch_size, input_dim).
            h_t (torch.Tensor): Previous hidden state tensor of shape (batch_size, hidden_dim).

        Returns:
            torch.Tensor: The updated hidden state tensor of shape (batch_size, hidden_dim).
        """
        batch_size = x_t.size(0)  # Get the batch size from the input tensor
        
        if h_t is None:
            h_t = self.initialize(x_t.size(0), x_t.device)

        # Assert statements to check the dimensions of inputs and hidden states
        assert x_t.size(1) == self.input_dim, f'Input dimension was expected to be {self.input_dim}, got {x_t.size(1)}'
        assert h_t.size(0) == batch_size, f'0th dimension of h_t is expected to be {batch_size}, got {h_t.size(0)}'
        assert h_t.size(1) == self.hidden_dim, f'Hidden dimension was expected to be {self.hidden_dim}, got {h_t.size(1)}' 
        
        # Calculate the new hidden state using tanh activation function
        h_t = torch.tanh(self.i2h(x_t) + self.h2h(h_t))
        
        # Assert statements to check the dimensions of the output hidden state
        assert h_t.size(0) == batch_size, f'0th dimension of output of RNNManualCell is expected to be {batch_size}, got {h_t.size(0)}'
        assert h_t.size(1) == self.hidden_dim, f'1st dimension of output of RNNManualCell is expected to be {self.hidden_dim}, got {h_t.size(1)}'
        
        return h_t  # Return the updated hidden state
    
    def initialize(self, batch_size, device=None):
        """
        Initialize the hidden state.

        Args:
            batch_size (int): The size of the batch.
            device (torch.device, optional): The device on which the tensor should be allocated. Defaults to None.

        Returns:
            torch.Tensor: The initialized hidden state tensor filled with zeros of shape (batch_size, hidden_dim).
        """
        return torch.zeros(batch_size, self.hidden_dim, device=device)  # Return a zero tensor for the initial hidden state
