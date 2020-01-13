"""Trains a neural net to play Pong.
   Exit using Ctrl-C. This will automatically save the neural network into file net-<random string>.
"""
import sys
import gym
import argparse
import torch
import baselines.common.atari_wrappers as atari_wrappers
import torch.optim as optim
from model import Net
from pong_env import PongEnvWrapper
from dql import DeepQLearner


parser = argparse.ArgumentParser(description='Train a neural net to play pong.')
parser.add_argument('--checkpoint', default="", help='file containing the training checkpoint')
parser.add_argument('--render', default=False, action='store_true', help='Render a screen to see the net play')


def train_pong(args):
    render = args['render']
    checkpoint = args['checkpoint']

    env = gym.make('PongNoFrameskip-v4')
    env = atari_wrappers.FireResetEnv(env)
    env = atari_wrappers.NoopResetEnv(env)
    env = PongEnvWrapper(env)
    action_space = [1,2,3]

    pong_learner = DeepQLearner(env, action_space, checkpoint=checkpoint, render=render)
    pong_learner.train(1000)

    env.close()

args = parser.parse_args()
args = {
    'checkpoint': args.checkpoint,
    'render': args.render,
}
train_pong(args)