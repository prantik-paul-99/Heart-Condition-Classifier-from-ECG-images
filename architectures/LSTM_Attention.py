class TemporalAttn(nn.Module):
    def __init__(self, hidden_size):
        super(TemporalAttn, self).__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.fc2 = nn.Linear(self.hidden_size*2, self.hidden_size, bias=False)

    def forward(self, hidden_states):
        score_first_part = self.fc1(hidden_states)
        h_t = hidden_states[:,-1,:]
        score = torch.bmm(score_first_part, h_t.unsqueeze(2)).squeeze(2)
        attention_weights = F.softmax(score, dim=1)
        context_vector = torch.bmm(hidden_states.permute(0,2,1), attention_weights.unsqueeze(2)).squeeze(2)
        pre_activation = torch.cat((context_vector, h_t), dim=1)
        attention_vector = self.fc2(pre_activation)
        attention_vector = torch.tanh(attention_vector)

        return attention_vector, attention_weights

    
class AttnLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(AttnLSTM, self).__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True)
        self.attn = TemporalAttn(hidden_size=hidden_size)
        self.fc = nn.Linear(hidden_size, 5)

    def forward(self, x):
        x, (h_n, c_n) = self.lstm(x)
        x, weights = self.attn(x)
        x = self.fc(x)
        return x
