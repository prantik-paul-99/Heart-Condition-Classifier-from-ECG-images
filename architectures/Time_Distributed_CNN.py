class TimeDistributed(nn.Module):
    def __init__(self, module, time_steps, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        
        self.layers = nn.ModuleList([module for i in range(time_steps)])
        
        self.time_steps = time_steps
        self.batch_first = batch_first

    def forward(self, x):

        batch_size, time_steps, channels, H, W = x.size()
        output = torch.tensor([]).to("cuda:0")
        
        for i in range(time_steps):
            output_t = self.layers[i](x[:, i, :, :, :])
            output_t  = output_t.unsqueeze(1)
            output = torch.cat((output, output_t), 1)
        
        return output


class Time_Distributed_CNN_Network(nn.Module):
    def __init__(self):
        super(Time_Distributed_CNN_Network, self).__init__()


        self.time_steps = 4
        self.extraction_layers = nn.Sequential(
            TimeDistributed(module=nn.LazyConv2d(out_channels=64, kernel_size=3, padding=1), 
                            time_steps=self.time_steps),
            nn.ReLU(),
            TimeDistributed(module=nn.MaxPool2d(kernel_size=2, stride=2), time_steps=self.time_steps),
            
            TimeDistributed(module=nn.LazyConv2d(out_channels=128, kernel_size=3, padding=1), 
                            time_steps=self.time_steps),
            nn.ReLU(),
            TimeDistributed(module=nn.MaxPool2d(kernel_size=2, stride=2), time_steps=self.time_steps),
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