hyperparameters = {
    'batch_size':[32, 128, 256, 512],
    'dropout':[0, 0.1, 0.2,0.5],
    'random_state': [42],
    'optimizer_kwargs' : [{"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-2}],
    'hidden_size': [64, 128, 256],
    'n_rnn_layers': [1, 2, 3],
    'hidden_fc_sizes': [[64], [64, 64], [128, 128]]
}
 
