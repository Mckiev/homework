
import os
import pickle
import tensorflow as tf
import numpy as np
import pandas as pd
import tf_util
import gym
import load_policy
from BC import *
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    parser.add_argument('dim', type=int)
    parser.add_argument('--render', action='store_true',
                        help='Whether to render a data aggregation rollout')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_aggr', type=int, default=10,
                        help='Number of data aggregation iterations')
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of test runs of policy for evaluation')
    args = parser.parse_args()

    with tf.Session():
        tf_util.initialize()

        expert = load_policy.load_policy('experts/' + args.envname + '.pkl')
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        #Getting starting samples from one rollout of the expert policy
        observations, actions, _ = rollout(expert, env, max_steps, render=False)
        #Training the initial model
        model, _ = train_model(args.dim, observations, actions, 2)
        test_data = test_model(model, env, num_rollouts=args.num_rollouts)
        returns = pd.DataFrame({**test_data, 'Aggregations' : 0, 'Observations' : len(observations)})

        #Data aggregation and retraining
        for i in range(args.num_aggr):
            print('Aggregation ', i+1)
            new_obs, act, totalr = rollout(model.predict, env, max_steps, render=args.render)
            exp_actions = np.array([expert(obs[None,:]) for obs in new_obs])
            exp_actions = exp_actions.reshape(len(exp_actions),-1)
            observations = np.concatenate((observations, new_obs))
            actions = np.concatenate((actions, exp_actions))

            model, _ = train_model(args.dim, observations, actions, 2, model = model)
            test_data = test_model(model, env, num_rollouts=args.num_rollouts)

            new_data = pd.DataFrame({**test_data, 'Aggregations' : i+1, 'Observations' : len(observations)})
            returns = returns.append(new_data, sort=False)

        returns["Algorithm"] = 'DAgger'
        returns.to_csv('DAggr_'+args.envname+'-'+str(args.dim)+'.csv')

