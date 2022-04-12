# Best hyperparameters for stochastic tp with 100 samples
hyperparameters = {
    "model" : "LSTM",
    "batch_size" : 512,
    "dropout" : 0,
    "random_state" : 42,
    "optimizer_kwargs" : {"lr" : 1.0e-2},
    "hidden_dim" : 64,
    "n_rnn_layers" : 1,
    "n_epochs" : 500,
    "model_name" : "stochastic_100_0",
    "log_tensorboard" : False,
    "input_chunk_length" : 25,
    "output_chunk_length" : 24,
    "force_reset" : True,
    "save_checkpoints" : True,
    "torch_device_str" : "cuda:0",
}
