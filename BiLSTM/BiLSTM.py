import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first = True, 
            bidirectional = True
        )
        self.fc = nn.Linear(hidden_size*2, num_classes)  # 隐层包含向前层和向后层两层，所以隐层共有两倍的Hidden_size
    
    def forward(self, x):
        # 初始话LSTM的隐层和细胞状态
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device) # 同样考虑向前层和向后层
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        
        # 前向传播 LSTM
        out, _ = self.lstm(x, (h0, c0))  # LSTM输出大小为 (batch_size, seq_length, hidden_size*2)
        
        # 解码最后一个时刻的隐状态
        out = self.fc(out[:, -1, :])
        return out