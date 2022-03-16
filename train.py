from lstm import LSTM
from utils import sliding_windows, process_data, x_space
from model import tipping_point
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu" 

# Defining variables for the LSTM
num_epochs = 2000
learning_rate = 0.01

input_size = 1
hidden_size = 64
num_layers = 1

viz_reps = 1

in_seq_length = 25
out_seq_length = 10

lstm = LSTM(out_seq_length, input_size, hidden_size, num_layers, dev)
lstm.to(dev)

# Getting data in right format
_data = tipping_point()
training_data = _data.collect_samples(1000)

criterion = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
trainXs, trainYs = process_data(training_data, in_seq_length, out_seq_length)

for epoch in range(num_epochs):
    # Note this data loading is not efficient, since i'm doing this every loop
    # Should do outside of the loop and draw an index instead -- TO DO
    idx = np.random.randint(0, len(training_data))
    trainX = trainXs[idx].to(dev)
    trainY = trainYs[idx].to(dev)

    outputs = lstm(trainX)
    optimizer.zero_grad()
    
    # obtain the loss function
    loss = criterion(outputs, trainY)
    
    loss.backward()
    
    optimizer.step()
    if epoch % 100 == 0:
      print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

# Evaluate
lstm.eval()

test_data = _data.collect_samples(100)
testXs, testYs = process_data(test_data, in_seq_length, out_seq_length)
for i in range(viz_reps):
    idx = np.random.randint(0, len(test_data))
    testX = testXs[idx].to(dev)
    testY = testYs[idx].to(dev)
    
    train_predict = lstm(testX)
    x_linspace = x_space(testY.shape[0], testY.shape[1])
    data_predict = train_predict.data.cpu().numpy()
    dataY_plot = testY.data.cpu().numpy()
    
    plt.plot(x_linspace, dataY_plot, color="b", alpha=0.1)
    plt.plot(x_linspace, data_predict, color="orange", alpha=0.7)
    
plt.savefig("trash")
