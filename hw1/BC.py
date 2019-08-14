import os, json
import pickle
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import gym

def save_params(path, params):
    '''Saves parametes to the path as json
    creates a desired folder if it not exists
    prints out the path upon completion
    '''
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(path, 'w') as out:
        out.write(json.dumps(params, separators=(',\n','\t:\t'), sort_keys=True))
    print ('Data saved to ' + path)

def rollout(policy_fn, env, max_steps = 1000, render = False):
    '''Rolls out a policy on an environment until copletion of the episode
    returns : observations, actions, total reward
    '''
    observations, actions = [],[]
    obs = env.reset()
    done = False
    totalr = 0.
    steps = 0
    while not done:
        action = policy_fn(obs[None,:])
        observations.append(obs)
        actions.append(action)
        obs, r, done, _ = env.step(action)
        totalr += r
        steps += 1
        if render:
            env.render()
        if steps >= max_steps:
            break

    return np.array(observations).reshape(len(observations),-1), np.array(actions).reshape(len(actions),-1), totalr

def train_model(hidden_size, obs, act, patience, model = None):
    '''
    Trains a model of 2 layer of specified size
    on provided observation-action samples with
    given the patience of early stopping
    each epoch will contain 20k samples(created by repeating)
    returns model and some training data
    '''
    if not model:
        model = tf.keras.Sequential([
        layers.Dense(hidden_size, activation='sigmoid', input_shape=(obs.shape[1],)),
        layers.Dense(hidden_size, activation='sigmoid'),
        layers.Dense(act.shape[1])])

        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                      loss='mse',
                      metrics=['mse'])

    es = tf.keras.callbacks.EarlyStopping(patience=patience, monitor='loss')

    dataset = tf.data.Dataset.from_tensor_slices((obs, act))
    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=500)
    dataset = dataset.batch(32)
    
    history = model.fit(dataset, epochs=1000, steps_per_epoch=625, callbacks=[es])

    training_data = dict(hidden_size = hidden_size, epochs = es.stopped_epoch, mse = history.history['loss'][-1], patience = patience)

    return model, training_data


def test_model(model, env , num_rollouts = 20, max_steps = 1000):
    '''
    Tests model on an environment for the specified number
    of rollouts and returns the test results
    '''
    returns = []
    obs_set = []
    act_set = []
    for i in range(num_rollouts):
        print('Rollout ', i+1)
        o, a, totalr = rollout(model.predict, env, max_steps)
        returns.append(totalr)

    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))

    test_data = dict(Returns = returns)
    
    return  test_data 
    

if __name__ == '__main__':

    ### This code will train the Behavioral Cloning agent on the expert data
    ### And test it for the provided number of runs. Data will be saved in
    ### BC_models folder

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    parser.add_argument('hidden_size', type = int, default = 256)
    parser.add_argument('--patience', type=int, default=3,
                        help='Patience of early stopping. Default = 3')
    parser.add_argument('--num_tests', type=int, default=20,
                        help='Number test runs of the learned policy. Default = 20')
    args = parser.parse_args()

    # Getting the expert policy samples
    with open(os.path.join('expert_data', args.envname + '.pkl'), 'rb') as f:
        expert_data = pickle.load(f)
    obs, act = expert_data.values()
    obs, act = obs.reshape(len(obs), -1) , act.reshape(len(act), -1)

    model_path = os.path.join('BC_models', args.envname+ '_' + str(args.hidden_size)+'.h5')
    data_path = os.path.join('BC_models', args.envname+ '_' + str(args.hidden_size)+'.json')

    model, training_data = train_model(args.hidden_size, obs, act, args.patience)
    training_data["model_path"] = model_path

    if args.num_tests:
        env = gym.make(args.envname)
        test_data = test_model(model, env, num_rollouts = args.num_tests)
        training_data.update(test_data)

    save_params(data_path, training_data)
    model.save(model_path)





