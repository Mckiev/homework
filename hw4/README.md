# CS294-112 HW 4: Model Based RL

This is a completed homework number 4 from CS294-112 (2018)

The results can be reproduced by running: bash run_all.sh \
Problem statements : [HW4.pdf](hw4.pdf) \
Homework report : [HW4_report.pdf](HW4_report.pdf) 


#### PROBLEM 1

In every dimension except 17th the predictions of the model are almost exactly the same as the ground truth. In state dimension #17 predictions are getting much different from the truth : small errors accumulated over time because each successive state was predicted based on erroneous previous state

![P1](data/HalfCheetah_q1_exp/prediction_005.jpg)

#### PROBLEM 2

Comparison of the performance of the model based controller trained on a random data versus the random policy:

Reward / Policy |Model Based Controller | Random Policy
 --- | --- | ---
ReturnAvg | -162.79|28.00
ReturnStd |-11.47 | 19.30

#### PROBLEM 3(A)


Model Based RL policy return increases with iterations 

![P3a](plots/HalfCheetah_q3_default.jpg)

#### PROBLEM 3(B)


Comparing performance for varying [number of nn layers, length of MPC horizon, number of ranom actions]  
![P3b1](plots/HalfCheetah_q3_nn_layers.jpg)
![P3b2](plots/HalfCheetah_q3_mpc_horizon.jpg)
![P3b3](plots/HalfCheetah_q3_actions.jpg)