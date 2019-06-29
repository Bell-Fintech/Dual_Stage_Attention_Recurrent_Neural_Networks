import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as tf
import time
from constants import device

def init_hidden(x, hidden_size: int):
    """
    Train the initial value of the hidden state:
    https://r2rt.com/non-zero-initial-states-for-recurrent-neural-networks.html
    """
    return Variable(torch.zeros(1, x.size(0), hidden_size))


class Encoder(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, T: int):
        """
        input size: number of underlying factors (81) #number of features at each time step
        T: number of time steps (10) in the encoder
        hidden_size: dimension of the hidden state i.e., no of nodes to be used for the cell state as well as the hidden state of the LSTM
        """
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.T = T

        self.lstm_layer = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1)
        self.attn_linear = nn.Linear(in_features=2 * hidden_size + T - 1, out_features=1)

    def forward(self, input_data):
        # input_data: (batch_size, T - 1, input_size)


        input_weighted = Variable(torch.zeros(input_data.size(0), self.T - 1, self.input_size))
        input_encoded = Variable(torch.zeros(input_data.size(0), self.T - 1, self.hidden_size))

        # hidden, cell: initial states with dimension hidden_size
        hidden_state = init_hidden(input_data, self.hidden_size)  # 1 * batch_size * hidden_size
        cell_state = init_hidden(input_data, self.hidden_size)
        # input_in_cpu = input_data.cpu()#-----------------------------------------------------------------------------------------and here

        for t in range(self.T - 1):


            #Note:
            # x = torch.tensor([1, 2, 3])
            # x.repeat(4, 2)
            # give tensor([[ 1,  2,  3,  1,  2,  3],
            #         [ 1,  2,  3,  1,  2,  3],
            #         [ 1,  2,  3,  1,  2,  3],
            #         [ 1,  2,  3,  1,  2,  3]])
            #
            #
            #
            # View changes how the tensor is represented. For ex: a tensor with 4 elements can be represented as 4X1 or 2X2 or 1X4 but permute changes the axes.
            # While permuting the data is moved but with view data is not moved but just reinterpreted.


            # Eqn. 8: concatenate the hidden states with each predictor
            x = torch.cat((hidden_state.cuda().repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           cell_state.cuda().repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           input_data.permute(0, 2, 1)), dim=2)  # batch_size * input_size * (2*hidden_size + T - 1)#---------changed here

            x = x.cuda()
            # Eqn. 8: Get attention weights
            x = self.attn_linear(x.view(-1, self.hidden_size * 2 + self.T - 1))  # (batch_size * input_size) * 1
            # Eqn. 9: Softmax the attention weights
            attn_weights = tf.softmax(x.view(-1, self.input_size), dim=1)  # (batch_size, input_size)
            # Eqn. 10: LSTM
            weighted_input = torch.mul(attn_weights, input_data[:, t, :])  # (batch_size, input_size)
            # Fix the warning about non-contiguous memory
            # see https://discuss.pytorch.org/t/dataparallel-issue-with-flatten-parameter/8282
            self.lstm_layer.flatten_parameters()

            # print(weighted_input)
            # print("\n")
            # print(hidden)
            # print("\n")
            # print(cell)
            # time.sleep(333)

            hidden_state = hidden_state.to(device)#-------------------changed to gpu here
            cell_state = cell_state.to(device)#-------------------changed to gpu here

            _, lstm_states = self.lstm_layer(weighted_input.unsqueeze(0), (hidden_state, cell_state))
            hidden_state = lstm_states[0]
            cell_state = lstm_states[1]
            # Save output
            input_weighted[:, t, :] = weighted_input
            input_encoded[:, t, :] = hidden_state

        return input_weighted, input_encoded


class Decoder(nn.Module):

    def __init__(self, encoder_hidden_size: int, decoder_hidden_size: int, T: int, out_feats=1):
        super(Decoder, self).__init__()

        self.T = T
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        self.attn_layer = nn.Sequential(nn.Linear(2 * decoder_hidden_size + encoder_hidden_size,
                                                  encoder_hidden_size),
                                        nn.Tanh(),
                                        nn.Linear(encoder_hidden_size, 1))
        self.lstm_layer = nn.LSTM(input_size=out_feats, hidden_size=decoder_hidden_size)
        self.fc = nn.Linear(encoder_hidden_size + out_feats, out_feats)
        self.fc_final = nn.Linear(decoder_hidden_size + encoder_hidden_size, out_feats)

        self.fc.weight.data.normal_()

    def forward(self, input_encoded, y_history):
        input_encoded = input_encoded.to(device)
        # input_encoded: (batch_size, T - 1, encoder_hidden_size)
        # y_history: (batch_size, (T-1))
        # Initialize hidden and cell, (1, batch_size, decoder_hidden_size)
        hidden = init_hidden(input_encoded, self.decoder_hidden_size).to(device)
        cell = init_hidden(input_encoded, self.decoder_hidden_size).to(device)
        context = Variable(torch.zeros(input_encoded.size(0), self.encoder_hidden_size))

        for t in range(self.T - 1):
            # (batch_size, T, (2 * decoder_hidden_size + encoder_hidden_size))
            x = torch.cat((hidden.repeat(self.T - 1, 1, 1).permute(1, 0, 2),
                           cell.repeat(self.T - 1, 1, 1).permute(1, 0, 2),
                           input_encoded), dim=2)
            x = x.to(device)
            # Eqn. 12 & 13: softmax on the computed attention weights
            x = tf.softmax(
                    self.attn_layer(
                        x.to(device).view(-1, 2 * self.decoder_hidden_size + self.encoder_hidden_size)
                    ).view(-1, self.T - 1),
                    dim=1)  # (batch_size, T - 1)

            # Eqn. 14: compute context vector
            context = torch.bmm(x.unsqueeze(1), input_encoded)[:, 0, :]  # (batch_size, encoder_hidden_size)

            # Eqn. 15
            y_tilde = self.fc(torch.cat((context, y_history[:, t]), dim=1))  # (batch_size, out_size)
            # Eqn. 16: LSTM
            self.lstm_layer.flatten_parameters()
            _, lstm_output = self.lstm_layer(y_tilde.unsqueeze(0), (hidden, cell))
            hidden = lstm_output[0]  # 1 * batch_size * decoder_hidden_size
            cell = lstm_output[1]  # 1 * batch_size * decoder_hidden_size

        # Eqn. 22: final output
        return self.fc_final(torch.cat((hidden[0], context), dim=1))
