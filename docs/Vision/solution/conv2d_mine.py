import torch
import torch.nn as nn 
import torch.nn.functional as F 

from typing import Tuple 

class Conv2d_Mine(nn.Module):
    def __init__(
        self, 
        kernel_size: int, 
        stride: int = 1, 
        padding: int = 2, 
    ):
        super(Conv2d_Mine, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding 

        self.weight: nn.Parameter = nn.Parameter(
            torch.randn(
                kernel_size, 
                kernel_size,
            )
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        if self.padding != 0:
            x = F.pad(
                x, 
                pad = (
                    self.padding, 
                    self.padding, 
                    self.padding, 
                    self.padding
                ), 
                mode = 'constant', 
                value = 0, 
            )
        batch_size, input_h, input_w = x.shape 
        kernel_h, kernel_w = self.kernel_size, self.kernel_size 
        stride_h, stride_w = self.stride, self.stride

        output_h = (input_h - kernel_h) // stride_h + 1 
        output_w = (input_w - kernel_w) // stride_w + 1 

        output = torch.zeros(
            batch_size, output_h, output_w
        )

        for batch_idx in range(batch_size):
            for h in range(output_h):
                for w in range(output_w):
                    h_start = h * stride_h 
                    h_end = h_start + kernel_h 
                    w_start = w * stride_w
                    w_end = w_start + kernel_w 
                    input = x[batch_idx, h_start:h_end, w_start:w_end]
                    output[batch_idx, h, w] = torch.sum(input * self.weight)
        
        return output 

class CNN(nn.Module):
    def __init__(self, output_labels):
        super(CNN, self).__init__() 
        self.conv_layer_r = Conv2d_Mine(5, 1, 2)
        self.conv_layer_g = Conv2d_Mine(5, 1, 2)
        self.conv_layer_b = Conv2d_Mine(5, 1, 2)
        # (config.batch_size, 32, 32)
        # (32 + 2 + 2 - 5) // 1 + 1 = 32
        self.fc_r = nn.Linear(1024, 30)
        self.fc_g = nn.Linear(1024, 30) 
        self.fc_b = nn.Linear(1024, 30) 

        self.final_fc = nn.Linear(90, 10)

    def forward(self, x):
        batch_size = x.size(0)
        r = x[:, 0] 
        g = x[:, 1]
        b = x[:, 2]

        r = self.conv_layer_r(r) 
        g = self.conv_layer_g(g)
        b = self.conv_layer_b(b) 

        r = F.relu(self.fc_r(r.view(batch_size, -1)))
        g = F.relu(self.fc_g(g.view(batch_size, -1)))
        b = F.relu(self.fc_b(b.view(batch_size, -1)))

        return self.final_fc(torch.concat((r, g, b), dim = 1))
        # return self.final_fc(r) 

if __name__ == '__main__':
    from data_handler import train_data, train_loader

    cnn = CNN(train_data.classes)

    criterion = nn.CrossEntropyLoss()

    for batch_x, batch_y in train_loader:
        t = cnn(batch_x)
        print(t.shape)
        loss = criterion(t, batch_y)
        print(loss.item())
        break 

