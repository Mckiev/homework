import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('logdir', type=str)
args = parser.parse_args()


with open(args.logdir, 'rb') as f:
	returns, lengths = pickle.load(f)

running_av_ret =[np.mean(returns[i:i+100]) for i in range(len(returns)-100)] 

plt.plot(np.cumsum(lengths)[100:],running_av_ret)
plt.show()