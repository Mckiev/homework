import os, json
import pickle
import tensorflow as tf
import numpy as np
import pandas as pd
import gym
from BC import test_model, save_params, train_model

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('envname', type=str)
parser.add_argument('hidden_size', type = int, default = 256)
parser.add_argument('--range', nargs='+', type=int, default = [1000,5000,5],
					help='''Range of trainig samples for defferent models to train on
							format - 3 numbers separated with spaces:  start, stop, n_samples.
							Default: 1000, 5000, 5. It amounts to 5 experiments on 1000, 2000,...,5000 samples ''')
parser.add_argument('--num_tests', type=int, default=20,
                    help='Number test runs of the learned policy. Default = 20')
parser.add_argument('--patience', type=int, default=3,
                    help='Patience of early stopping')

args = parser.parse_args()


with open(os.path.join('expert_data', args.envname + '.pkl'), 'rb') as f:
	expert_data = pickle.load(f)

obs, act = expert_data.values() 
obs, act = obs.reshape(len(obs), -1) , act.reshape(len(act), -1)
env = gym.make(args.envname)


dirname  = 'BC_vs_obs/' + args.envname+'_' + str(args.hidden_size)



results = pd.DataFrame()

for n_samples in np.linspace(*args.range, dtype = int):

	model, training_data = train_model(args.hidden_size, obs[:n_samples], act[:n_samples], args.patience)
	test_data = test_model(model, env, num_rollouts = args.num_tests)

	model_path = os.path.join(dirname, str(n_samples)+'.h5')
	data_path = os.path.join(dirname, str(n_samples)+'.json')
	test_data["Observations"] = int(n_samples)
	training_data["Observations"] = int(n_samples)
	training_data["model_path"] = model_path
	save_params(data_path, training_data)
	model.save(model_path)
	
	results = results.append(pd.DataFrame(test_data), sort=False)

res_path = os.path.join(dirname, 'results.csv')
results["Algorithm"] = 'BC'
results.to_csv(res_path)

	
