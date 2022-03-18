import numpy as np
from torch.autograd import Variable
import torch
import os

def sliding_windows(data, in_seq_length, out_seq_length):
    x = []
    y = []
    
    data = data.reshape(-1)
    for i in range(len(data)-in_seq_length-out_seq_length):
        _x = data[i:(i+in_seq_length)]
        _y = data[i+in_seq_length:i+in_seq_length+out_seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)

def process_data(data, in_seq_length, out_seq_length):
    x_list = []
    y_list = []
    
    for i, t_series in enumerate(data):
        x, y = sliding_windows(t_series, in_seq_length, out_seq_length)
        x = x.reshape(-1, in_seq_length, 1)
        y = y.reshape(-1, out_seq_length)

        trainX = Variable(torch.Tensor(x))
        trainY = Variable(torch.Tensor(y))
        x_list.append(trainX)  
        y_list.append(trainY)
    
    return x_list, y_list

def x_space(input_window, output_window, n_rows):
    x = [[i for i in range(j + input_window, j + input_window + output_window)] for j in range(n_rows)]
    return x

def check_make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
