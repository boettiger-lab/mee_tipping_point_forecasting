hyperparameters = {
    'batch_size':[1, 8, 16, 32, 64, 128, 256],
    'dropout':[0,0.2,0.5],
    'random_state': [42],
    'optimizer_kwargs' : [{"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}, {"lr": 1e-2}],
    'nhead': [4, 8, 12, 16],
    'num_encoder_layers': [1, 2],
    'num_decoder_layers': [1, 2],
}

