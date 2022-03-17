from lstm import LSTM
from utils import sliding_windows, process_data, x_space
from model import tipping_point
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu" 

parser = argparse.ArgumentParser()
parser.add_argument(
    "-e",
    "--epochs",
    default=2000,
    type=int,
    help="# of Epochs",
)
parser.add_argument(
    "--lr",
    default=0.01,
    type=float,
    help="Learning rate",
)
parser.add_argument(
    "-f",
    "--num_features",
    default=1,
    type=int,
    help="# of features",
)
parser.add_argument(
    "-s",
    "--hidden_layers",
    default=1,
    type=int,
    help="Number of Hidden Layers",
)
parser.add_argument(
    "-w",
    "--width_hidden",
    default=64,
    type=int,
    help="Width of Hidden Layers",
)
parser.add_argument(
    "-i",
    "--input_window",
    default=25,
    type=int,
    help="Input window size for the model",
)
parser.add_argument(
    "-p",
    "--prediction_window",
    default=25,
    type=int,
    help="Input window size for the LSTM",
)
parser.add_argument(
    "-v",
    "--viz_iter",
    default=1,
    type=int,
    help="# replicates to examine for visualization",
)
parser.add_argument(
    "-n",
    "--plot_name",
    default="figure",
    type=str,
    help="Name of the output plot",
)
args = parser.parse_args()

input_size = 1
num_layers = 1

lstm = LSTM(args.prediction_window, input_size, args.width_hidden, args.hidden_layers, dev)
lstm.to(dev)

# Getting data in right format
_data = tipping_point()
training_data = _data.collect_samples(1000)

criterion = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(lstm.parameters(), lr=args.lr)
trainXs, trainYs = process_data(training_data, args.input_window, args.prediction_window)

for epoch in range(args.epochs):
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
testXs, testYs = process_data(test_data, args.input_window, args.prediction_window)
for i in range(args.viz_iter):
    idx = np.random.randint(0, len(test_data))
    testX = testXs[idx].to(dev)
    testY = testYs[idx].to(dev)
    
    train_predict = lstm(testX)
    x_linspace = x_space(testY.shape[0], testY.shape[1])
    data_predict = train_predict.data.cpu().numpy()
    dataY_plot = testY.data.cpu().numpy()
    
    plt.plot(x_linspace, dataY_plot, color="b", alpha=0.1)
    plt.plot(x_linspace, data_predict, color="orange", alpha=0.5)
    
plt.savefig(f"{args.plot_name}")
