#!/bin/bash
# Need to evaluate hopf from training atm! this is due to data scaler not being portable 
# (I could write a function here to save training set but time is short rn)

python evaluate.py -f lstm --sim_model saddle -s 1 -o lstm_saddle_1_nontipped
python evaluate.py -f block_rnn --sim_model saddle -s 1 -o block_saddle_1_nontipped
python evaluate.py -f gru --sim_model saddle -s 1 -o gru_saddle_1_nontipped
python evaluate.py -f transformer --sim_model saddle -s 1 -o transformer_saddle_1_nontipped

python evaluate.py -f lstm --sim_model saddle -s 10 -o lstm_saddle_10_nontipped
python evaluate.py -f block_rnn --sim_model saddle -s 10 -o block_saddle_10_nontipped
python evaluate.py -f gru --sim_model saddle -s 10 -o gru_saddle_10_nontipped
python evaluate.py -f transformer --sim_model saddle -s 10 -o transformer_saddle_10_nontipped

python evaluate.py -f lstm --sim_model stochastic -s 1 -o lstm_stochastic_1_tipped --tipped
python evaluate.py -f block_rnn --sim_model stochastic -s 1 -o block_stochastic_1_tipped --tipped
python evaluate.py -f gru --sim_model stochastic -s 1 -o gru_stochastic_1_tipped --tipped
python evaluate.py -f transformer --sim_model stochastic -s 1 -o transformer_stochastic_1_tipped --tipped

python evaluate.py -f lstm --sim_model stochastic -s 10 -o lstm_stochastic_10_tipped --tipped
python evaluate.py -f block_rnn --sim_model stochastic -s 10 -o block_stochastic_10_tipped --tipped
python evaluate.py -f gru --sim_model stochastic -s 10 -o gru_stochastic_10_tipped --tipped
python evaluate.py -f transformer --sim_model stochastic -s 10 -o transformer_stochastic_10_tipped --tipped