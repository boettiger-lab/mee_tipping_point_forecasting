#!/bin/bash
python tune_special_cases.py -c tipped --seed 42 --prefix stochastic_tipped_sc
python tune_special_cases.py -c non_tipped --seed 42 --prefix stochastic_nontipped_sc
python tune_special_cases.py -c both --seed 42 --prefix stochastic_both_sc

python tune_special_cases.py -t saddle -c tipped --seed 42 --prefix stochastic_tipped_sc
python tune_special_cases.py -t saddle -c non_tipped --seed 42 --prefix stochastic_nontipped_sc
python tune_special_cases.py -t saddle -c both --seed 42 --prefix stochastic_both_sc

python tune.py -s 10 --seed 42 --prefix stochastic_10_samples
python tune.py -s 10 -t saddle --seed 42 --prefix saddle_10_samples

python tune.py -s 100 --seed 42 --prefix stochastic_100_samples
python tune.py -s 100 -t saddle --seed 42 --prefix saddle_100_samples

python tune.py -s 1000 --seed 42 --prefix stochastic_1000_samples
python tune.py -s 1000 -t saddle --seed 42 --prefix saddle_1000_samples