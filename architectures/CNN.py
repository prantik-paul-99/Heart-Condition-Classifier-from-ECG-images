class HybridNetwork(nn.Module):
    def __init__(self):
        super(HybridNetwork, self).__init__()

        self.extraction_layers = nn.Sequential(
            nn.LazyConv2d(out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.LazyConv2d(out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classification_layers = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.LazyLinear(64),
            nn.ReLU(),
            nn.LazyLinear(5),
            nn.LogSoftmax(dim=1)
        )


    def forward(self, input_X):
        
        for layer in self.extraction_layers:
            input_X = layer(input_X)
        
        for layer in self.classification_layers:
            input_X = layer(input_X)

        return input_X