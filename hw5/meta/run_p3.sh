#!/usr/bin/env bash

python3 train_policy.py 'pm' --exp_name P3_gran5 --history 10 -lr 5e-4 --discount 0.9 -rec --gran 5 -ep 20
python3 train_policy.py 'pm' --exp_name P3_gran10 --history 10 -lr 5e-4 --discount 0.9 -rec --gran 10 -ep 20
python3 train_policy.py 'pm' --exp_name P3_gran2 --history 10 -lr 5e-4 --discount 0.9 -rec --gran 2 -ep 20