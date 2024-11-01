import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        hidden_channels: int = 16,
        output_channels: int = 10,
        # num_classes: int = 10
    ):
        super(SimpleCNN, self).__init__()
        self.conv1: nn.Conv2d = nn.Conv2d(
            in_channels = input_channels,
            out_channels = hidden_channels,
            kernel_size = 3,
            padding = 1,
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.relu: nn.ReLU = nn.ReLU()
        self.pool: nn.MaxPool2d = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv2: nn.Conv2d = nn.Conv2d(
            in_channels = hidden_channels,
            out_channels = hidden_channels,
            kernel_size = 3,
            padding = 1
        )
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(
            in_channels = hidden_channels,
            out_channels = hidden_channels,
            kernel_size = 3,
            padding = 1
        )
        self.bn3 = nn.BatchNorm2d(64)

        self.dropout = nn.Dropout(p=0.5)
        # self.fc: nn.Linear = nn.Linear(hidden_channels * 8 * 8, output_channels)
        self.fc: nn.Linear = nn.Linear(hidden_channels * 4 * 4, output_channels)
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        # x: (batch_size, 3, 32, 32)
        batch_size = x.size(0)

        x = self.conv1(x) # (batch_size, 16, 32, 32)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x) # (batch_size, 16, 16, 16)

        x = self.conv2(x) # (batch_size, 32, 16, 16)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x) # (batch_size, 32, 8, 8)

        x = self.conv3(x) # (batch_size, 64, 8, 8)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x) # (batch, 64, 4, 4)

        x = x.view(batch_size, -1) # (batch_size, 64 * 4 * 4)
        # (batch_size, 32 * 8 * 8)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class ResNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        hidden_channels: int = 16,
        output_channels: int = 10,
        depth: int = 4,
        device = None
    ):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels = input_channels,
            out_channels = hidden_channels,
            kernel_size = 3,
            padding = 1,
            stride = 1,
        )
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(p=0.5)

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels= hidden_channels, 
                    out_channels = hidden_channels,
                    kernel_size = 3,
                    padding = 1,
                    stride = 1,
                ),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU()
            )for _ in range (depth)
        ])
        self.fc = nn.Linear(hidden_channels * 32 * 32, output_channels)

        self.device = device
        self.to(device)
    
    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.bn1(x)
        before = self.relu(x)

        for layer in self.layers:
            layer.to(self.device)
            after = layer(before)
            next_before = before + after
            before = next_before

        after = after.view(batch_size, -1)
        after = self.dropout(after)

        return self.fc(after)