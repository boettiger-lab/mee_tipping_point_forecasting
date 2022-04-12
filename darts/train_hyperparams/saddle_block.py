# Best hyperparameters for stochastic tp with 100 samples
hyperparameters = {
    "model" : "LSTM",
    "batch_size" : 32,
    "dropout" : 0.1,
    "random_state" : 42,
    "optimizer_kwargs" : {"lr" : 1.0e-3},
    "hidden_size" : 64,
    "hidden_fc_sizes": [128, 128],
    "n_rnn_layers" : 1,
    "n_epochs" : 500,
    "model_name" : "saddle_block",
    "log_tensorboard" : False,
    "input_chunk_length" : 25,
    "output_chunk_length" : 24,
    "force_reset" : True,
    "save_checkpoints" : True,
    "torch_device_str" : "cuda:0",
}
