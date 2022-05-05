#!/bin/bash

python StatsForecastAutoARIMA.py -s 1 -o arima_hopf_1_increase --sim_model hopf
python StatsForecastAutoARIMA.py -s 1 -o arima_stochastic_1_tipped --sim_model stochastic --tipped
python StatsForecastAutoARIMA.py -s 1 -o arima_saddle_1_nontipped --sim_model saddle