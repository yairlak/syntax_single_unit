from torch import nn

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
