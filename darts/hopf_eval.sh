#!/bin/bash
python train_ensemble.py -f lstm -s 1 -o lstm_hopf_1_increase -e --sim_model hopf
python train_ensemble.py -f block_rnn -s 1 -o block_hopf_1_increase -e --sim_model hopf
python train_ensemble.py -f gru -s 1 -o gru_hopf_1_increase  -e --sim_model hopf
python train_ensemble.py -f transformer -s 1 -o tranformer_hopf_1_increase  -e --sim_model hopf

python train_ensemble.py -f lstm -s 10 -o lstm_hopf_10_increase  -e --sim_model hopf
python train_ensemble.py -f block_rnn -s 10 -o block_hopf_10_increase -e --sim_model hopf
python train_ensemble.py -f gru -s 10 -o gru_hopf_10_increase -e --sim_model hopf
python train_ensemble.py -f transformer -s 10 -o tranformer_hopf_10_increase -e --sim_model hopf