import math

# from pandas import read_csv
# import numpy as np
import torch
from torch import nn

# from torch.utils.data import DataLoader
# from torchvision import datasets
# from torchvision.transforms import v2
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error
# import matplotlib.pyplot as plt


class ManualRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.hidden_size = hidden_size

        # Maps the current input x_t to the hidden dimension
        self.x_to_h = nn.Linear(input_size, hidden_size)

        # Maps the previous hidden state h_{t-1} to the hidden dimension
        self.h_to_h = nn.Linear(hidden_size, hidden_size)

        # Maps the final hidden state to the prediction
        self.h_to_y = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, time_steps, input_size)

        batch_size = x.size(0)
        time_steps = x.size(1)

        # Initial hidden state h_0
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)  # type: ignore

        # Process sequence one time step at a time
        for t in range(time_steps):
            x_t = x[:, t, :]  # shape: (batch_size, input_size)

            h = torch.tanh(self.x_to_h(x_t) + self.h_to_h(h))  # type: ignore

        # Use final hidden state for prediction
        y = self.h_to_y(h)

        return y
