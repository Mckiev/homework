# CS294-112 HW 3: Q-Learning

This is a completed homework number 3 from CS294-112 (2018)\
I have implemented the DQN and Actor-Critic algorithms. The starter code was provided\
The instructions can be seen in [hw3.pdf](hw3.pdf) \
My report is in [HW3_report.pdf](HW3_report.pdf)

To reproduce the results from the first part of the assignment run p1.sh \
To make make a sweep of model architectures run : python3 model_sweep.py (should take a while to run) \
For the second part of the assigment run p2.sh \


### Original Readme 
Dependencies:
 * Python **3.5**
 * Numpy version **1.14.5**
 * TensorFlow version **1.10.5**
 * MuJoCo version **1.50** and mujoco-py **1.50.1.56**
 * OpenAI Gym version **0.10.5**
 * seaborn
 * Box2D==**2.3.2**
 * OpenCV
 * ffmpeg

Before doing anything, first replace `gym/envs/box2d/lunar_lander.py` with the provided `lunar_lander.py` file.

The only files that you need to look at are `dqn.py` and `train_ac_f18.py`, which you will implement.

See the [HW3 PDF](http://rail.eecs.berkeley.edu/deeprlcourse/static/homeworks/hw3.pdf) for further instructions.

The starter code was based on an implementation of Q-learning for Atari generously provided by Szymon Sidor from OpenAI.
