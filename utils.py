import numpy as np
from torch.autograd import Variable
import torch

def sliding_windows(data, seq_length):
    x = []
    y = []
    
    data = data.reshape(-1)
    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)

def process_data(data, seq_length):
    x_list = []
    y_list = []
    
    for i, t_series in enumerate(data):
        x, y = sliding_windows(t_series, seq_length)
        x = x.reshape(-1, seq_length, 1)
        y = y.reshape(-1, 1)

        trainX = Variable(torch.Tensor(x))
        trainY = Variable(torch.Tensor(y))
        x_list.append(trainX)  
        y_list.append(trainY)
    
    return x_list, y_list
