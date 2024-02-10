from torch import nn
import torch
import torch.nn.functional as F

class LSTMBasedFusion(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTMBasedFusion, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Tomar la salida de la última secuencia
        x = self.fc(x)
        return x


class HybridLSTMCNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(HybridLSTMCNN, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.conv1 = nn.Conv1d(in_channels=hidden_size, out_channels=64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x.permute(0, 2, 1)  # Reorganizar para las dimensiones de convolución
        x = F.relu(self.conv1(x))
        x = torch.mean(x, dim=2)  # Agregación de características
        x = self.fc(x)
        return x


class TrainableChannelFusion(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(TrainableChannelFusion, self).__init__()
        self.attention_weights = nn.Parameter(torch.randn(input_size))
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        weights = F.softmax(self.attention_weights, dim=0)
        x = x * weights
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x


class TrainableChannelFusion2(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=1, bidirectional=False, dropout_rate=0.0):
        super(TrainableChannelFusion2, self).__init__()
        self.attention_weights = nn.Parameter(torch.randn(input_size))
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                            num_layers=num_layers, batch_first=True, 
                            bidirectional=bidirectional, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, num_classes)

    def forward(self, x):
        weights = F.softmax(self.attention_weights, dim=0)
        x = x * weights
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x