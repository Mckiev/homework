#!/bin/bash

python3 run_dqn_atari.py --doubleQ --subdir DQ_compare
python3 run_dqn_atari.py --subdir DQ_compare

python3 plot.py Logz/DQ_compare --pic_name DQ_compare.png