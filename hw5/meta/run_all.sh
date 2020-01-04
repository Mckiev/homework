#Problem 1
python train_policy.py 'pm-obs' --exp_name P1 􏰁→ --history 1 -lr 5e-5 -n 200 --num_tasks 4

#Problem 2
python3 train_policy.py 'pm' --exp_name DEF_P2_h60_ff --history 60 -lr 5e-4 --discount 0.9 -n 60 -s 32
python3 train_policy.py 'pm' --exp_name DEF_P2_h20_ff --history 20 -lr 5e-4 --discount 0.9 -n 60 -s 64
python3 train_policy.py 'pm' --exp_name DEF_P2_h10_ff --history 10 -lr 5e-4 --discount 0.9 -n 60 -s 64 -l 2 
python3 train_policy.py 'pm' --exp_name DEF_P2_h5_ff --history 5 -lr 5e-4 --discount 0.9 -n 60 -s 90 

python3 train_policy.py 'pm' --exp_name DEF_P2_h60_rec --history 60 -lr 5e-4 --discount 0.9 -n 60 -rec 
python3 train_policy.py 'pm' --exp_name DEF_P2_h20_rec --history 20 -lr 5e-4 --discount 0.9 -n 60 -rec 
python3 train_policy.py 'pm' --exp_name DEF_P2_h10_rec --history 10 -lr 5e-4 --discount 0.9 -n 60 -rec 
python3 train_policy.py 'pm' --exp_name DEF_P2_h5_rec --history 5 -lr 5e-4 --discount 0.9 -n 60 -rec 
python3 train_policy.py 'pm' --exp_name DEF_P2_h3_rec --history 3 -lr 5e-4 --discount 0.9 -n 60 -rec
python3 train_policy.py 'pm' --exp_name DEF_P2_h2_rec --history 2 -lr 5e-4 --discount 0.9 -n 60 -rec

#Problem 3
python3 train_policy.py 'pm' --exp_name P3_gran5 --history 10 -lr 5e-4 --discount 0.9 -rec --gran 5  -n 60
python3 train_policy.py 'pm' --exp_name P3_gran10 --history 10 -lr 5e-4 --discount 0.9 -rec --gran 10  -n 60
python3 train_policy.py 'pm' --exp_name P3_gran2 --history 10 -lr 5e-4 --discount 0.9 -rec --gran 2 -n 60
