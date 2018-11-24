import torch

def skew(tensor):
    # BCHW to BCH(H+W-1)
    output = tensor.new_zeros((tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[2] + tensor.shape[3] - 1))
    for row in range(tensor.shape[2]):
        columns = (row, row + tensor.shape[3])
        output[:, :, row, columns[0]:columns[1]] = tensor[:, :, row]
    return output


def unskew(tensor):
    # BCH(H+W-1) to BCHW
    output = tensor.new_zeros((tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[3] - tensor.shape[2] + 1))
    for row in range(tensor.shape[2]):
        columns = (row, row + output.shape[3])
        output[:, :, row] = tensor[:, :, row, columns[0]:columns[1]]
    return output