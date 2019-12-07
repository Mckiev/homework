#!/bin/bash

python3 train_ac_f18.py CartPole-v0 -n 100 -b 1000 -e 3 -ntu 10 -ngsptu 10 --exp_name n10_10 --subdir CP_data
python3 train_ac_f18.py CartPole-v0 -n 100 -b 1000 -e 3 -ntu 100 -ngsptu 1 --exp_name n100_1 --subdir CP_data
python3 train_ac_f18.py CartPole-v0 -n 100 -b 1000 -e 3 -ntu 1 -ngsptu 1 --exp_name n1_1 --subdir CP_data
python3 train_ac_f18.py CartPole-v0 -n 100 -b 1000 -e 3 -ntu 1 -ngsptu 100 --exp_name n1_100 --subdir CP_data

python3 plot.py CP_data --x Iteration --value AverageReturn --pic_name CP_sweep.png
 
python3 train_ac_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b 30000 -lr 0.02 --exp_name HC10_10 -ntu 10 -ngsptu 10 --subdir HC_data
python3 plot.py HC_data --x Iteration --value AverageReturn --pic_name HC.png


python3 train_ac_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.95 -n 100 -e 3 -l 2 -s 64 -b 5000 -lr 0.01 --exp_name IP10_10  -ntu 10 -ngsptu 10 --subdir IP_data
python3 plot.py IP_data --x Iteration --value AverageReturn --pic_name IP.png