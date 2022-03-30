hyperparameters = {
    'batch_size':[1, 16, 64, 128, 256, 512],
    'dropout':[0,0.2,0.5],
    'random_state': [42],
    'optimizer_kwargs' : [{"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}, {"lr": 1e-2}],
    'hidden_dim': [64, 128, 256],
    'n_rnn_layers': [1, 2, 3],
}
 
