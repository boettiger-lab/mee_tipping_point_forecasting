# Best hyperparameters for saddle node tp with 100 samples
hyperparameters = {
    "model" : "LSTM",
    "batch_size" : 1,
    "dropout" : 0.2,
    "random_state" : 42,
    "optimizer_kwargs" : {"lr" : 1.0e-4},
    "hidden_dim" : 64,
    "n_rnn_layers" : 2,
    "n_epochs" : 200,
    "model_name" : "saddle_100",
    "log_tensorboard" : False,
    "input_chunk_length" : 25,
    "output_chunk_length" : 24,
    "force_reset" : True,
    "save_checkpoints" : True,
    "torch_device_str" : "cuda:0",
}
