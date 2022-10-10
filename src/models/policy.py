"""Module summary.

This is policy.py.
"""

from torch import nn
import torch
import numpy as np

class policy_lstm(nn.Module):
    """Summary line.

    This type policy (motion estimator) utilizes (non-convolutional) LSTM.
    
    """

    def __init__(self, pc_size):
        super().__init__()

        # lstm seq2seq model for the "Motion Estimator" component
        INPUT_DIM = (pc_size**2) * 3 # The input is (X,Y,R)
        OUTPUT_DIM = (pc_size**2) * 2 # The output is (X,Y)
        HID_DIM = 512
        N_LAYERS = 3
        DROPOUT = 0.5

        # Initialize lstm network.
        self.lstm = LSTMcell(INPUT_DIM, OUTPUT_DIM, HID_DIM, N_LAYERS, DROPOUT)
        # initialize weights
        for name, param in self.lstm.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)

    def forward(self, input):
        """sum 2 values.

        Args:
            x (float): 1st argument
            y (float): 2nd argument

        Returns:
            float: summation

        """
        
        bsize, tsize, channels, height, width = input.size()

        # Lagrangian prediction
        R_grd = input[:,0,:,:,:] #use initial


class LSTMcell(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.rnn = nn.LSTM(input_dim, hid_dim, n_layers, dropout = dropout,
                           batch_first=True)

        # linear layer for converting hidden dim to output dim
        self.fc_out = nn.Linear(hid_dim,output_dim)

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, in_seq):
            
        #in_seq = [in_seq len, batch size]
        
        in_seq = self.dropout(in_seq)
        
        rnn_out, (hidden, cell) = self.rnn(in_seq)
        
        #outputs = [in_seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]

        outputs = self.fc_out(rnn_out)
        
        #outputs are always from the top hidden layer

        return outputs
