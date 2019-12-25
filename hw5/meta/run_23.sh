#!/usr/bin/env bash

#Problem 2
python3 train_policy.py 'pm-far' --exp_name P2_h20_ff --history 20 -lr 5e-4 --discount 0.9 -s 64 -ep 100
python3 train_policy.py 'pm-far' --exp_name P2_h20_rec --history 20 -lr 5e-4 --discount 0.9 -rec -ep 100 

python3 train_policy.py 'pm-far' --exp_name P2_h10_ff --history 10 -lr 5e-4 --discount 0.9 -s 64 -l 2 -ep 100
python3 train_policy.py 'pm-far' --exp_name P2_h10_rec --history 10 -lr 5e-4 --discount 0.9 -rec -ep 100

python3 train_policy.py 'pm-far' --exp_name P2_h5_ff --history 5 -lr 5e-4 --discount 0.9 -s 90 -ep 100
python3 train_policy.py 'pm-far' --exp_name P2_h5_rec --history 5 -lr 5e-4 --discount 0.9 -rec -ep 100

python3 train_policy.py 'pm-far' --exp_name P2_h3_rec --history 3 -lr 5e-4 --discount 0.9 -rec -ep 100



python3 train_policy.py 'pm' --exp_name DEF_P2_h60_ff --history 60 -lr 5e-4 --discount 0.9 -s 32

python3 train_policy.py 'pm' --exp_name DEF_P2_h20_ff --history 20 -lr 5e-4 --discount 0.9 -s 64

python3 train_policy.py 'pm' --exp_name DEF_P2_h10_ff --history 10 -lr 5e-4 --discount 0.9 -s 64 -l 2

python3 train_policy.py 'pm' --exp_name DEF_P2_h5_ff --history 5 -lr 5e-4 --discount 0.9 -s 90

python3 train_policy.py 'pm' --exp_name DEF_P2_h60_rec --history 60 -lr 5e-4 --discount 0.9 -rec 

python3 train_policy.py 'pm' --exp_name DEF_P2_h20_rec --history 20 -lr 5e-4 --discount 0.9 -rec

python3 train_policy.py 'pm' --exp_name DEF_P2_h10_rec --history 10 -lr 5e-4 --discount 0.9 -rec

python3 train_policy.py 'pm' --exp_name DEF_P2_h5_rec --history 5 -lr 5e-4 --discount 0.9 -rec



python3 train_policy.py 'pm' --exp_name P3_gran5 --history 10 -lr 5e-4 --discount 0.9 -rec --gran 5 
python3 train_policy.py 'pm' --exp_name P3_gran10 --history 10 -lr 5e-4 --discount 0.9 -rec --gran 10 
python3 train_policy.py 'pm' --exp_name P3_gran2 --history 10 -lr 5e-4 --discount 0.9 -rec --gran 2 

# #Problem 3
# python3 train_policy.py 'pm-far' --exp_name P3_gran10 --history 10 -lr 5e-4 --discount 0.9 -rec --gran 10 -ep 100
# python3 train_policy.py 'pm-far' --exp_name P3_gran_max --history 10 -lr 5e-4 --discount 0.9 -rec --gran -10 -ep 100
