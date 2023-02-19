import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()

        # build LeNet-5 using nn.Sequential
        self.model = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Flatten(start_dim=1),
            nn.LazyLinear(120),
            nn.ReLU(),
            nn.LazyLinear(84),
            nn.ReLU(),
            nn.LazyLinear(10),
            nn.LogSoftmax(dim=1)
        )


    def forward(self, input_X):
      # print(f'forward: {input_X.shape}')
      # return self.model(input_X)

      for layer in self.model:
        input_X = layer(input_X)
        # print(input_X.shape)

      return input_X
    