from skewers import skew, unskew
from pixel_lstm import DiagonalPixelLSTM
import torch
from random import randint
from functools import reduce

def test_input_row(debug=False):
    print("Testing PixelLSTM input convolution - Row")
    array = torch.tensor([[
        [
            [float('+inf'), float('+inf'), float('+inf'), float('+inf')],
            [float('-inf'), float('-inf'), float('-inf'), float('-inf')],
            [float('-inf'), float('-inf'), float('-inf'), float('-inf')],
        ]
    ]], dtype=torch.float32)
    if debug:
        print("Input:")
        print(array)
    with torch.no_grad():
        lstm = DiagonalPixelLSTM(1, 1)
        for name, parameter in lstm.state_dict().items():
            if name == 'conv_is.weight':
                parameter.fill_(1.0)
            else:
                parameter.fill_(0.0)
        output = lstm(array)
    if debug:
        print("Output:")
        print(output)
    target = torch.tanh(torch.tensor([[
        [
            [1.0, 2.0, 3.0, 4.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    ]], dtype=torch.float32))
    assert torch.all(output == target), "FAILED"
    print("DiagonalPixelLSTM Test PASSED")
    
def test_input_col(debug=False):
    print("Testing PixelLSTM input convolution - Col")
    array = torch.tensor([[
        [
            [float('+inf'), float('+inf'), float('+inf'), float('+inf')],
            [float('-inf'), float('-inf'), float('-inf'), float('-inf')],
            [float('-inf'), float('-inf'), float('-inf'), float('-inf')],
        ]
    ]], dtype=torch.float32).transpose(2, 3)
    if debug:
        print("Input:")
        print(array)
    with torch.no_grad():
        lstm = DiagonalPixelLSTM(1, 1)
        for name, parameter in lstm.state_dict().items():
            if name == 'conv_is.weight':
                parameter.fill_(1.0)
            else:
                parameter.fill_(0.0)
        output = lstm(array)
    if debug:
        print("Output:")
        print(output)
    target = torch.tanh(torch.tensor([[
        [
            [1.0, 2.0, 3.0, 4.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    ]], dtype=torch.float32)).transpose(2, 3)
    assert torch.all(output == target), "FAILED"
    print("DiagonalPixelLSTM Test PASSED")


def main():
    test_input_row()
    test_input_col(True)
    
if __name__ == "__main__":
    main()