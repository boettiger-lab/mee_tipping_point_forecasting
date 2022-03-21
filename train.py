from lstm import LSTM
from utils import sliding_windows, process_data, x_space, check_make_dir
from model import tipping_point
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import os

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu" 

parser = argparse.ArgumentParser()
parser.add_argument(
    "-r",
    "--var_reps",
    default=20,
    type=int,
    help="# of samples to use to calculate variance",
)
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
    "-l",
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
    default=75,
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
    "--model_name",
    default="model",
    type=str,
    help="Name of the output plot",
)
parser.add_argument(
    "-t",
    "--training_sample_size",
    default=1000,
    type=int,
    help="Number of samples to use in training",
)
parser.add_argument(
    "-s",
    "--save",
    action='store_true',
    help="Flag to save the model",
)
parser.add_argument(
    "-d",
    "--dropout",
    default=0.0,
    type=float,
    help="Flag to save the model",
)
args = parser.parse_args()

input_size = 1 # Dimension of input

# Firing up the LSTM
lstm = LSTM(args.prediction_window, input_size, args.width_hidden, args.hidden_layers, dev, args.dropout)
lstm.to(dev)

# Getting data in right format
_data = tipping_point()
training_data = _data.collect_samples(args.training_sample_size)

criterion = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(lstm.parameters(), lr=args.lr)

# Getting data windowed and in format acceptable to the LSTM
trainXs, trainYs = process_data(training_data, args.input_window, args.prediction_window)

for epoch in range(args.epochs):
    # Pick a random time series each epoch to train on
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

check_make_dir(f"evaluated_models/{args.model_name}")

if args.save:
    torch.save(lstm, f"evaluated_models/{args.model_name}/{args.model_name}")

# Evaluating the LSTM now

# Looking at different samples than we trained on
test_data = _data.collect_samples(100)
testXs, testYs = process_data(test_data, args.input_window, args.prediction_window)

for i in range(args.viz_iter):
    idx = np.random.randint(0, len(test_data))
    testX = testXs[idx].to(dev)
    testY = testYs[idx].to(dev)
    dataY_plot = testY.data.cpu().numpy()
    x_linspace = x_space(testX.shape[1], testY.shape[1], testY.shape[0])
    plt.plot(x_linspace, dataY_plot, color="b", alpha=1)
    
    for j in range(args.var_reps):
        # For dropout case, all we have to do is re-run however many replicates
        train_predict = lstm(testX)
        data_predict = train_predict.data.cpu().numpy()
        plt.plot(x_linspace, data_predict, color="orange", alpha=0.1)
        
    plt.savefig(f"evaluated_models/{args.model_name}/{args.model_name}_{i}")
    plt.clf()
