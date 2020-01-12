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
        ob = self.env.reset(**kwargs)
        ob = utils.rgb_to_grayscale(ob) # will return array of shape (210, 160)
        ob = torch.from_numpy(ob).unsqueeze(0) # add back the channel dim
        return ob

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        ob = utils.rgb_to_grayscale(ob) # will return array of shape (210, 160)
        ob = torch.from_numpy(ob).unsqueeze(0) # add back the channel dim
        return ob, reward, done, info
