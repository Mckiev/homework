import numpy as np
from gym import spaces
from gym import Env


class FarPointEnv(Env):
    """
    point mass on a 2-D plane
    goals are sampled randomly from a square
    """

    def __init__(self, num_tasks=1, is_evaluation = False, gran=1):
        self.reset_task(is_evaluation = is_evaluation, gran=gran)
        self.reset()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,))

    def reset_task(self, is_evaluation = False, gran=1):
        '''
        sample a new task randomly

        Problem 3: make training and evaluation goals disjoint sets
        if `is_evaluation` is true, sample from the evaluation set,
        otherwise sample from the training set
        '''
        #====================================================================================#
        #                           ----------PROBLEM 3----------
        #====================================================================================#
        def sample():
            x = np.random.uniform(-5,5)
            x += np.sign(x) * 5
            y = np.random.uniform(-5,5)
            y += np.sign(y) * 5
            return x,y

        x,y = sample()

        if is_evaluation:
            # If gran is negative make the goal one particular point 
            # far from any point from the training set
            if gran<0:
                x, y = -10, -10
            else:
                while not self.in_eval(x,y,gran):
                    x,y = sample()
        else:
            while self.in_eval(x,y,gran):
                x,y = sample()

        
        self._goal = np.array([x, y])

    def in_eval(self, x, y, gran):
        '''
        Checks if coordinates (x,y) belong to evaluation set or training set
        provided the gran level of granularity. The bigger gran the more 
        dissimilar are sets
        '''
        gran = int(gran)

        check = x//gran + y//gran
        return check % 2 == 0

         


    def reset(self):
        self._state = np.array([0, 0], dtype=np.float32)
        return self._get_obs()

    def _get_obs(self):
        return np.copy(self._state)

    def reward_function(self, x, y):
        return - (x ** 2 + y ** 2) ** 0.5

    def step(self, action):
        x, y = self._state
        # compute reward, add penalty for large actions instead of clipping them
        x -= self._goal[0]
        y -= self._goal[1]
        # action = np.clip(action, self.action_space.low, self.action_space.high)
        penalty = 0 if self.reward_function(*action) > -0.5 else -10*self.reward_function(*action)**2
        reward = self.reward_function(x, y) + penalty
        # check if task is complete
        done = reward > -0.1 
        # move to next state
        self._state = self._state + action
        ob = self._get_obs()
        return ob, reward, done, dict()

    def viewer_setup(self):
        print('no viewer')
        pass

    def render(self):
        print('current state:', self._state)

    def seed(self, seed):
        np.random.seed = seed


