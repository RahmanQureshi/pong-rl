"""Trains a neural net to play Pong.
   Exit using Ctrl-C. This will automatically save the neural network into file net-<random string>.
"""
import gym
import argparse
import torch
import baselines.common.atari_wrappers as atari_wrappers
import torch.optim as optim
from model import Net
from pong_env import PongEnvWrapper
from dql import DeepQLearner


parser = argparse.ArgumentParser(description='Train a neural net to play pong.')
parser.add_argument('--net_file', metavar='net_file', type=str, nargs=1, default="",
                    help='file to load the model to be trained')
parser.add_argument('--render', default=False, action='store_true',
                    help='Render a screen to see the net play')


def train_pong(args):
    net = args['net']
    optmizer = args['optimizer']
    render = args['render']

    env = gym.make('PongNoFrameskip-v4')
    env = atari_wrappers.FireResetEnv(env)
    env = atari_wrappers.NoopResetEnv(env)
    env = PongEnvWrapper(env)
    action_space = [1,2,3]

    pong_learner = DeepQLearner(net, optimizer, env, action_space, render=render)
    pong_learner.train(10)

    env.close()

args = parser.parse_args()
net_file = args.net_file[0] if len(args.net_file) == 1 else ''
net = Net(1)
optimizer = optim.RMSprop(net.parameters(), lr=0.0025, alpha=0.9, eps=1e-02, momentum=0.0)
if net_file != '':
    checkpoint = torch.load(net_file)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
args = {
    'net': net,
    'optimizer': optimizer,
    'render': args.render,
}
train_pong(args)