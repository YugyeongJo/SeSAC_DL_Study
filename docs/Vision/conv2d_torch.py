import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(
        self, 
        num_classes: int = 10, 
        device: torch.device = 'cpu',
    ):
        super(SimpleCNN, self).__init__()
        self.device = device
        self.conv1: nn.Conv2d = nn.Conv2d(
            in_channels=3, 
            out_channels=16, 
            kernel_size=3, 
            padding=1,
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.relu: nn.ReLU = nn.ReLU()
        self.pool: nn.MaxPool2d = nn.MaxPool2d(
            kernel_size = 2, 
            stride = 2
            )
        self.conv2: nn.Conv2d = nn.Conv2d(
            in_channels = 16, 
            out_channels = 32, 
            kernel_size=3, 
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3: nn.Conv2d = nn.Conv2d(
            in_channels = 32, 
            out_channels = 32, 
            kernel_size=3, 
            padding=1,
        )
        self.bn3 = nn.BatchNorm2d(32)
        self.dropout: nn.Dropout = nn.Dropout(p=0.5)
        self.fc: nn.Linear = nn.Linear(32 * 8 * 8, num_classes)
        
        self.to(self.device)
        
    def forward(self, x: torch.tensor) -> torch.tensor:
        x = x.to(self.device)
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
        x = self.conv3(x) # (batch_size, 32, 8, 8)
        x = self.bn3(x)
        x = x.view(batch_size, -1) # (batch_size, 32 * 8 * 8)
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
        device: torch.device = 'cpu',
    ):
        super(ResNet, self).__init__()
        self.device = device
        self.conv1 = nn.Conv2d(
            in_channels = input_channels, 
            out_channels = hidden_channels, 
            kernel_size = 3, 
            padding = 1, 
            stride = 1
        )
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.relu = nn.ReLU()
        
        self.dropout = nn.Dropout(p=0.5)
        # nn.ModuleList로 변경하여 각 레이어에 device 적용
        self.layers = nn.ModuleList([
            nn.Conv2d(
                in_channels = hidden_channels, 
                out_channels = hidden_channels, 
                kernel_size = 3, 
                padding = 1,
                stride = 1,
            )
            for _ in range(depth)
        ])
        
        self.bns = nn.ModuleList([
            nn.BatchNorm2d(hidden_channels) for _ in range(depth)
        ])
        
        self.fc = nn.Linear(hidden_channels * 32 * 32, output_channels)
        
        self.to(self.device)
        
    def forward(self, x):
        x = x.to(self.device)
        batch_size = x.size(0)
        x = self.conv1(x) # (batch_size, hidden_channels, 32, 32)
        x = self.bn1(x)
        before = self.relu(x)
        
        for layer, bn in zip(self.layers, self.bns):  # (batch_size, hidden_channels, 32, 32)
            after = layer(before)  # 각 layer의 출력을 device로 이동할 필요 없음 (이미 device로 이동된 상태)
            after = bn(after) 
            before = before + after
    
        after = after.view(batch_size, -1)  # (batch_size, hidden_channels * 32 * 32)
        after = self.dropout(after)
        return self.fc(after)