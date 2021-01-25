import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class RNNClassifier(nn.Module):
    def __init__(
            self,
            input_dim=10,
            output_dim=30,
            rec_layer_type='lstm',
            num_units=4,
            num_layers=1,
            dropout=0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rec_layer_type = rec_layer_type.lower()
        self.num_units = num_units
        self.num_layers = num_layers
        self.dropout = dropout

        rec_layer = {'lstm': nn.LSTM, 'gru': nn.GRU}[self.rec_layer_type]
        # We have to make sure that the recurrent layer is batch_first,
        # since sklearn assumes the batch dimension to be the first
        self.rec = rec_layer(self.input_dim, self.num_units, num_layers=num_layers, batch_first=True)
        self.output = nn.Linear(self.num_units, self.output_dim)

    def forward(self, X):
        # from the recurrent layer, only take the activities from the last sequence step
        if self.rec_layer_type == 'gru':
            _, rec_out = self.rec(X)
        else:
            _, (rec_out, _) = self.rec(X)
        rec_out = rec_out[-1]  # take output of last RNN layer
        drop = nn.functional.dropout(rec_out, p=self.dropout)
        # Remember that the final non-linearity should be softmax, so that our predict_proba
        # method outputs actual probabilities
        out = nn.functional.softmax(self.output(drop), dim=-1)
        return out


class CNN(nn.Module):
    def __init__(self,
                 num_timepoints=32,
                 in_channels=80,
                 out_channels=1,
                 kernel_size=5,
                 kernel_size_maxp=5,
                 stride=1,
                 stride_maxp=1,
                 dilation=1,
                 dilation_maxp=1,
                 num_classes=15,
                 dropout=0):

        super().__init__()
        # input: [batch_size, in_channels, num_timepoints]
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        d = (num_timepoints-dilation*(kernel_size-1)-1)/stride + 1
        # dropout
        self.conv1_drop = nn.Dropout(p=dropout)
        # input: [batch_size, out_channels, (num_timepoints-(kernel_size-1)-1)/stride + 1]
        self.maxpool1 = nn.MaxPool1d(kernel_size=kernel_size_maxp, stride=stride_maxp)
        d = (d-dilation_maxp*(kernel_size_maxp-1)-1)/stride_maxp + 1
        # input: [batch_size, out_channels, same-as-for-conv1d-wrt-new-input] 
        self.fc = nn.Linear(int(np.floor(d)*out_channels), num_classes) # output for previous layer is flatten, which gives multiplication here.
        self.fc_drop = nn.Dropout(p=dropout)

    def forward(self, x):
        #x = F.relu(self.maxpool1(self.conv1_drop(self.conv1(x))))
        x = F.relu(self.maxpool1(self.conv1(x)))
        x = x.flatten(1) # flatten the tensor starting at dimension 1
        x = self.fc_drop(self.fc(x))
        x = F.softmax(x)
        return x


class CNN2(nn.Module):
    def __init__(self,
                 num_timepoints=10,
                 in_channels=10,
                 out_channels=30,
                 kernel_size=3,
                 kernel_size_maxp=3,
                 stride=1,
                 stride_maxp=1,
                 dilation=1,
                 dilation_maxp=1,
                 num_classes=2,
                 dropout=0):

        super().__init__()
        # input: [batch_size, in_channels, num_timepoints]
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        d = (num_timepoints-dilation*(kernel_size-1)-1)/stride + 1
        # dropout
        self.conv1_drop = nn.Dropout(p=dropout)
        # input: [batch_size, out_channels, (num_timepoints-(kernel_size-1)-1)/stride + 1]
        self.maxpool1 = nn.MaxPool1d(kernel_size=kernel_size_maxp, stride=stride_maxp)
        d = (d-dilation_maxp*(kernel_size_maxp-1)-1)/stride_maxp + 1
        # 2nd layer
        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        d = (d-dilation*(kernel_size-1)-1)/stride + 1
        # dropout
        self.conv2_drop = nn.Dropout(p=dropout)
        # input: [batch_size, out_channels, (num_timepoints-(kernel_size-1)-1)/stride + 1]
        self.maxpool2 = nn.MaxPool1d(kernel_size=kernel_size_maxp, stride=stride_maxp)
        d = (d-dilation_maxp*(kernel_size_maxp-1)-1)/stride_maxp + 1
        # input: [batch_size, out_channels, same-as-for-conv1d-wrt-new-input] 
        self.fc = nn.Linear(int(np.floor(d)*out_channels), num_classes) # output for previous layer is flatten, which gives multiplication here.
        self.fc_drop = nn.Dropout(p=dropout)

    def forward(self, x):
        x = F.relu(self.maxpool1(self.conv1_drop(self.conv1(x))))
        x = F.relu(self.maxpool2(self.conv2_drop(self.conv2(x))))
        x = x.flatten(1) # flatten the tensor starting at dimension 1
        x = self.fc(x)
        x = F.softmax(x)
        return x


