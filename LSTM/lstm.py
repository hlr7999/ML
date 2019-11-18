import torch
from torch import nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()

        self.rnn = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True,
        )

        self.out = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)

        out = self.out(r_out[:,-1,:])
        return out
