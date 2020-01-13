import gym
import torch
import numpy as np
import utils
import baselines.common.atari_wrappers as atari_wrappers

class PongEnvWrapper(gym.Wrapper):
    """Pong specific wrapper which preprocesses the observations, etc.
    """

    def __init__(self, env):
        """
        """
        gym.Wrapper.__init__(self, env)


    def reset(self, **kwargs):
        ob = [self.env.reset(**kwargs)]
        noop_action = 0
        ob.append(self.env.step(noop_action)[0])
        ob = np.array([utils.rgb_to_grayscale(o[34:194][:][:]) for o in ob]) # will return array of shape (2, 210, 160)
        ob = torch.from_numpy(ob)
        return ob

    def step(self, action):
        ob = []
        total_reward = 0
        done = False
        info = None
        for i in range(0, 4):
            o, reward, done, info = self.env.step(action)
            total_reward = total_reward + reward 
            ob.append(o)
        ob = ob[-2:]
        ob = np.array([utils.rgb_to_grayscale(o[34:194][:][:]) for o in ob]) # will return array of shape (2, 210, 160)
        ob = torch.from_numpy(ob)
        if total_reward != 0: # Treat any reward as terminal. Terminal states have no observation (or don't need them anyway).
            info['done'] = True
        return ob, total_reward, done, info
