#!/usr/bin/env bash

python3 train_policy.py 'pm' --exp_name P2_h60_ff --history 60 -lr 5e-4 --discount 0.9 -s 32

python3 train_policy.py 'pm' --exp_name P2_h20_ff --history 20 -lr 5e-4 --discount 0.9 -s 64

python3 train_policy.py 'pm' --exp_name P2_h10_ff --history 10 -lr 5e-4 --discount 0.9 -s 64 -l 2

python3 train_policy.py 'pm' --exp_name P2_h10_ff --history 5 -lr 5e-4 --discount 0.9 -s 90

python3 train_policy.py 'pm' --exp_name P2_h60_rec --history 60 -lr 5e-4 --discount 0.9 -rec

python3 train_policy.py 'pm' --exp_name P2_h20_rec --history 20 -lr 5e-4 --discount 0.9 -rec

python3 train_policy.py 'pm' --exp_name P2_h10_rec --history 10 -lr 5e-4 --discount 0.9 -rec

python3 train_policy.py 'pm' --exp_name P2_h50_rec --history 5 -lr 5e-4 --discount 0.9 -rec