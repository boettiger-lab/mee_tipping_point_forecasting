# Best hyperparameters for stochastic tp with 100 samples
hyperparameters = {
    "model" : "LSTM",
    "batch_size" : 32,
    "dropout" : 0.5,
    "random_state" : 42,
    "optimizer_kwargs" : {"lr" : 1.0e-3},
    "hidden_size" : 128,
    "hidden_fc_sizes": [64, 64],
    "n_rnn_layers" : 2,
    "n_epochs" : 500, #1000
    "model_name" : "stochastic_100_0",
    "log_tensorboard" : False,
    "input_chunk_length" : 25,
    "output_chunk_length" : 24,
    "force_reset" : True,
    "save_checkpoints" : True,
    "torch_device_str" : "cuda:0",
}
