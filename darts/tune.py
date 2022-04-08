import numpy as np
import sys
sys.path.append("../")
from darts import TimeSeries
from darts.models import RNNModel, TCNModel, TransformerModel
from darts.utils.likelihood_models import LaplaceLikelihood
from utils import preprocessed_t_series
import random
import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument(
    "-m",
    "--model",
    default="lstm",
    type=str,
    help="model to train with (lstm, tcn or transformer)",
)
parser.add_argument(
    "-s",
    "--n_samples",
    default=100,
    type=int,
    help="# of samples to train on",
)
parser.add_argument(
    "--seed",
    default=42,
    type=int,
    help="Numpy random seed setting",
)
parser.add_argument(
    "--epochs",
    default=200,
    type=int,
    help="Number of epochs to train a model for",
)
parser.add_argument(
    "--device",
    default="cuda:0",
    type=str,
    help="Device to train the model on",
)
parser.add_argument(
    "--prefix",
    default="test",
    type=str,
    help="Prefix to use on model log dir",
)
parser.add_argument(
    "--n_trials",
    default=20,
    type=int,
    help="Number of tuning trials",
)
parser.add_argument(
    "-t",
    "--tp_model",
    default="stochastic",
    type=str,
    help="Select which model to use for the tipping point (stochastic/saddle)",
)
parser.add_argument(
    "--random_alpha",
    action="store_true",
    help="Select whether to use a random alpha or not on saddle node tp",
)
args = parser.parse_args()

np.random.seed(args.seed)
# Generating time series
train_series, _ = preprocessed_t_series(args.tp_model, args.n_samples, args.random_alpha)

if args.model == "lstm":
    from tune_hyperparams.lstm import hyperparameters
elif args.model == "tcn":
    from tune_hyperparams.tcn import hyperparameters
elif args.model == "transformer":
    from tune_hyperparams.transformer import hyperparameters
    
models = {"lstm": RNNModel, "tcn": TCNModel, "transformer": TransformerModel}

for i in range(args.n_trials):
    # Randomly selecting a parameter dictionary
    hyperparameter_keys = list(hyperparameters.keys())
    trial_hyperparameters = {
        hyperparameter_keys[j] : random.choice(hyperparameters[hyperparameter_keys[j]]) for j in range(len(hyperparameters))
    }
    
    # Making a file name suffix to keep track of hyperparams
    trial_hyperparameter_keys = list(trial_hyperparameters.keys())
    trial_string = f"{args.tp_model}"
    for k in range(len(hyperparameters)):
        trial_string += f"_{trial_hyperparameter_keys[k]}_{trial_hyperparameters[trial_hyperparameter_keys[k]]}"

    my_model = models[args.model.lower()](
        n_epochs=args.epochs,
        likelihood=LaplaceLikelihood(),
        model_name=f"{args.prefix}_{args.model}_tuning{trial_string}",
        log_tensorboard=True,
        input_chunk_length=25,
        output_chunk_length=24,
        torch_device_str = "cuda:0",
        **trial_hyperparameters
    )
    
    my_model.fit(
    train_series,
    verbose=True,
    )
