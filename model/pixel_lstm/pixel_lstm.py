import torch
import torch.nn as nn
from torch import sigmoid, tanh

from .skewers import skew, unskew

class DiagonalPixelLSTM(nn.Module):
    def __init__(self, channels_in, hidden_size):
        super(DiagonalPixelLSTM, self).__init__()
        self.channels_in = channels_in
        self.hidden_size = hidden_size
        self.conv_is = nn.Conv2d(channels_in, 5 * hidden_size, [1,1])
        self.conv_ss = nn.Conv1d(hidden_size, 5 * hidden_size, [2], padding=1)
        print("Created DiagonalPixelLSTM with parameters: {}".format({'channels_in': self.channels_in, 'hidden_size': self.hidden_size}))

    def forward(self, features):
        input_map = skew(features)

        # Precompute input to state transform as in Section 3.2 of Oord et al. 2016
        transformed_is = self.conv_is(input_map)

        hStates = []
        cStates = []

        previous_hidden_column = input_map.new_zeros([input_map.shape[0], self.hidden_size, input_map.shape[2]])
        previous_cell_column   = input_map.new_zeros([input_map.shape[0], self.hidden_size, input_map.shape[2]])

        for i in range(input_map.shape[3]):
            # Grab the ith column of the input_map
            input_column = input_map[..., i]

            # Implements Equation 3 in Oord et al. 2016
            # 'candidate_gate' is 'g' in the paper, a.k.a. the 'update'
            transformed_ss = self.conv_ss(previous_hidden_column)[..., :-1]
            gates = transformed_is[..., i] + transformed_ss
            output_gate, forget_gate_left, forget_gate_up, input_gate, candidate_gate = torch.chunk(gates, 5, dim=1)
            output_gate, forget_gate_left, forget_gate_up, input_gate = sigmoid(output_gate), sigmoid(forget_gate_left), sigmoid(forget_gate_up), sigmoid(input_gate)
            candidate_gate = tanh(candidate_gate)
            previous_cell_column_shifted = torch.cat([input_column.new_zeros([input_column.shape[0], self.hidden_size,  1]), previous_cell_column], 2)[..., :-1]
            next_cell_column = (forget_gate_left * previous_cell_column + forget_gate_up * previous_cell_column_shifted) + input_gate * candidate_gate
            next_hidden_column = output_gate * tanh(next_cell_column)

            hStates.append(next_hidden_column)
            cStates.append(next_cell_column)

            previous_hidden_column = next_hidden_column
            previous_cell_column = next_cell_column

        total_hStates = unskew(torch.stack(hStates, dim=3))
        total_cStates = unskew(torch.stack(cStates, dim=3))

        return total_hStates

