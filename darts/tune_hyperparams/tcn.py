hyperparameters = {
    'batch_size':[1, 8, 16, 32,64,128, 256],
    'dropout':[0,0.2,0.5],
    'random_state': [42],
    'optimizer_kwargs' : [{"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}, {"lr": 1e-2}],
    'dilation_base': [2, 3, 4],
    'num_layers': [None, 1, 2, 3],
    'kernel_size' : [2, 3, 4]
}
