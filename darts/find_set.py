import numpy as np
from utils import preprocessed_t_series, count_tipped
import argparse


flag_non = False
flag_tip = False
while True:
    x = preprocessed_t_series("stochastic", 1)
    y = x.all_values()
    
    if count_tipped(y) == 0:
        x.to_csv("stochastic_non_tipped.csv")
        flag_non = True

    if count_tipped(y) == 1:
        x.to_csv("stochastic_tipped.csv")
        flag_tip = True
    
    if flag_non and flag_tip:
        break
