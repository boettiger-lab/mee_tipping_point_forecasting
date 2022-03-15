from lstm import LSTM
from utils import sliding_windows
from model import tipping_point
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable

# Defining variables for the LSTM
num_epochs = 2000
learning_rate = 0.01

input_size = 1
hidden_size = 64
num_layers = 1

num_classes = 1

lstm = LSTM(num_classes, input_size, hidden_size, num_layers)

# Getting data in right format
_data = tipping_point()
training_data = _data.collect_samples(1)


seq_length = 4
x, y = sliding_windows(training_data, seq_length)
x = x.reshape(-1, 4, 1)
y = y.reshape(-1, 1)
dataX = Variable(torch.Tensor(x))
dataY = Variable(torch.Tensor(y))

trainX = Variable(torch.Tensor(x))
trainY = Variable(torch.Tensor(y))

criterion = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
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
train_predict = lstm(dataX)

data_predict = train_predict.data.numpy()
dataY_plot = dataY.data.numpy()

plt.plot(dataY_plot)
plt.plot(data_predict)
plt.suptitle('Time-Series Prediction')
plt.savefig("trash")
