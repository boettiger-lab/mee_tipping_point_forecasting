#!/bin/bash

for i in {0..10}
do
  python train.py -n ensemble_lstm_$i -s
done
