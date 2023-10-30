import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self):
        pass

class cnn_block(nn.Module):
    def __init__(self, filter_size):
        super().__init__()
        self.cnn = nn.Conv2d(1, 64, filter_size)
        self.bn = nn.BatchNorm()
        self.pool = nn.MaxPool2d(1, 2)
    
    def forward(self, x):
        x = self.cnn(x)
        x = self.bn(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        return x



class LSTM(nn.Module):
    
    def __init__(self):
        super().__init__()
    
    def forward(self):
        pass