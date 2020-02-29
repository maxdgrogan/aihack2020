import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, window, input_size=1, hidden_layer_size=100, 
                 num_of_layers=1, batch_size=8, output_size=1, dropout=0.2):
        super().__init__()
        self.batch_size = batch_size
        self.num_of_layers = num_of_layers
        self.hidden_layer_size = hidden_layer_size
        self.dropout = dropout
        self.window = window

        if len(input_size)<=1:
            self.input_size = input_size[0]
        else:
            self.input_size = input_size
        
        if len(output_size)<=1:
            self.output_size = output_size[0]
        else:
            self.output_size = output_size
        

        self.lstm = nn.LSTM(input_size = self.input_size,
                            hidden_size = self.hidden_layer_size,
                            num_layers = self.num_of_layers,
                            dropout = self.dropout, batch_first=True)

        self.linear = nn.Linear(self.hidden_layer_size, self.output_size)

        self.hidden_cell = [torch.randn(self.num_of_layers,self.batch_size,self.hidden_layer_size),
                            torch.randn(self.num_of_layers,self.batch_size,self.hidden_layer_size)]

    def reset(self, samples=None):
        if samples is None:
            samples = self.batch_size
        self.hidden_cell = [torch.zeros(self.num_of_layers,samples,self.hidden_layer_size),
                            torch.zeros(self.num_of_layers,samples,self.hidden_layer_size)]

    def forward(self, input_seq):

        batch_size, seq_len, _ = input_seq.size()
        self.reset(batch_size)

        self.hidden_cell[0] = self.hidden_cell[0].to(input_seq.device)
        self.hidden_cell[1] = self.hidden_cell[1].to(input_seq.device)

        lengths = torch.tensor([self.window for _ in range(batch_size)])

        input_seq = torch.nn.utils.rnn.pack_padded_sequence(input_seq, lengths, batch_first=True)
        
        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        lstm_out = lstm_out.contiguous()
        lstm_out = lstm_out[:,-1,:]

        predictions = self.linear(lstm_out)
        return predictions