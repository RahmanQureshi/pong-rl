"""Trains a neural net to play Pong.
   Exit using Ctrl-C. This will automatically save the neural network into file net-<random string>.
"""
import gym
from time import sleep
from model import Net
import copy
import torch
import numpy as np
import torch.optim as optim
import os
import psutil
import matplotlib.pyplot as plt
import random
import string
import sys
from signal import signal, SIGINT
import argparse
import baselines.common.atari_wrappers as atari_wrappers


parser = argparse.ArgumentParser(description='Train a neural net to play pong.')
parser.add_argument('--net_file', metavar='net_file', type=str, nargs=1, default="",
                    help='file to load the model to be trained')
parser.add_argument('--render', default=False, action='store_true',
                    help='Render a screen to see the net play')


class Experience:


    def __init__(self, state, action, result_state, reward):
        self.state = state
        self.action = action
        self.result_state = result_state
        self.reward = reward


def rgb_frame_to_grayscale(frame):
    """Convert rgb to grayscale. Uses observed luminance to compute grey value.
    """
    rgb = np.array([0.299, 0.587, 0.114])
    return np.inner(rgb, frame).astype(np.uint8)


def preprocess_frame(frame):
    """Applies a preprocess step to 
    """
    frame = frame[34:193, :] # just the play area
    return rgb_frame_to_grayscale(frame)


def print_memory_usage(device):
    process = psutil.Process(os.getpid())
    print("Memory usage (mb): {0}".format(process.memory_info().rss/1e6))
    if device.type == 'cuda':
        pass
        # Found that GPU usage is largely negligible (< 200mb)
        #print("GPU Memory Usage (mb): {0}".format(torch.cuda.memory_allocated(device)/1e6))


def random_string(stringLength=10):
    """Generate_sa random string of fixed length """
    lettersAndNumbers = string.ascii_lowercase + string.digits
    return ''.join(random.choice(lettersAndNumbers) for i in range(stringLength))


def save_net(net, optimizer):
    torch.save({
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, './net-' + random_string())


class CircleBuffer:
    """CircleBuffer implements a circular buffer. Also implements buffer access methods.
    """

    def __init__(self, size):
        self.size = size
        self.buffer = []
        self.position = 0 # tracks the oldest element in the array


    def push(self, e):
        if len(self.buffer) < self.size:
            self.buffer.append(None)
        self.buffer[self.position] = e
        self.position = (self.position + 1) % self.size


    def sample(self, n):
        return np.random.choice(self.buffer, n)

class DeepQLearner:


    def __init__(self, net, optimizer, env, action_space, replay_buffer_size=10000):
        self.net = net
        self.target_net = copy.deepcopy(self.net)
        self.replay_buffer = CircleBuffer(replay_buffer_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = optimizer
        self.lossfct = torch.nn.SmoothL1Loss()
        self.env = env
        self.epsilons = np.linspace(1,0.02,100000) # epsilon annealment. uses the last value after they are all used.
        self.num_env_steps = 0 # trakcs the number of times env.step() was called
        self.num_learning_steps = 0 # tracks the number of times learn() was called
        self.start_learning_iteration = 1000
        self.target_network_update_frequency = 1000 # after learning begins, update the target network after this many iterations
        self.action_space = action_space # index of neural network output correspond to these actions
        self.episode_rewards = []
        self.discount = 0.99


    def Q(self, net, x):
        """ Wrapper around NN forward which turns the tensor to a float type and moves it onto the device.
            Returns the value of all actions given the state.
        """
        return net(x.float().to(self.device))


    def max_target_Q(self, x):
        """Returns max Q(s,a) over all actions for the target neural network.
        """
        output = self.Q(self.target_net, x)
        return output.max().item()


    def get_epsilon(self):
        if self.iteration < self.start_learning_iteration: # before learning, use first epsilon value
            return epsilons[0]
        elif self.iteration-self.start_learning_iteration < len(self.epsilons): # annealment
            return epsilons[self.iteration-self.start_learning_iteration]
        # after running out of epsilons, continue to use the last one
        return epsilons[-1]


    def epsilon_greedy_action(self, x):
        epsilon = self.get_epsilon
        if np.random.uniform(0,1) <= epsilon: # with probability epsilon, pick random action
            return np.random.choice(self.action_space)
        return self.action_space[self.Q(x).argmax().item()]


    def deep_copy_nets(self):
        """Copies parameters of net into target_net
        """
        params1 = self.net.named_parameters()
        params2 = self.target_net.named_parameters()
        dict_params2 = dict(params2)
        for name1, param1 in params1:
            dict_params2[name1].data.copy_(param1.data)


    def save_plots(self):
        plt.figure(0)
        plt.title('Reward v.s. Episode #')
        plt.plot(self.episode_rewards)
        plt.show(block=False)
        plt.pause(0.1)


    def learn(self, minibatch_size=32):
        """DeepQLearner will look over all its experiences and do a single learning iteration step.
        """
        minibatch = self.replay_buffer.sample(minibatch_size)
        # construct the target values
        targets = [e.reward if e.result_state == None else e.reward + self.discount*self.max_target_Q(e.result_state) for e in minibatch]
        # compute the predicted Q values of all the experiences (state and actions taken) in the batch
        batch_states = torch.tensor([e.state for e in minibatch]) 
        predicted_Q_values = [q[minibatch[i].action] for (i, q) in enumerate(self.Q(self.net, batch_states))]
        # backward propagation
        optimizer.zero_grad()
        loss = self.lossfct(predicted_Q_values, targets)
        print("Loss: {}".format(loss))
        loss.backward()
        optimizer.step()
        self.num_learning_steps = self.num_learning_steps + 1
        if self.num_learning_steps % self.target_network_update_frequency == 0:
            self.deep_copy_nets()


    def train(self, num_episodes):
        for i in range(0, num_episodes):
            observation = env.reset()
            done = False
            episode_reward = 0
            while not done:
                action = self.epsilon_greedy_action(observation)
                result_observation, reward, done, info = env.step(action)
                self.replay_buffer.push(Experience(observation, action, result_observation, reward))
                observation = result_observation
                self.num_env_steps = self.num_env_steps + 1
                episode_reward = episode_reward + reward
                self.learn()
                print_memory_usage(self.device)
            # Episode is done. Save current status and plots.
            self.episode_rewards.append(episode_reward)
            self.save_plots()
            save_net(self.net, self.optimizer)


def train(args):
    net = args['net']
    optmizer = args['optimizer']
    render = args['render']

    env = gym.make('PongNoFrameskip-v4')
    env = atari_wrappers.FireResetEnv(env)
    env = atari_wrappers.NoopResetEnv(env)
    action_space = [1,2,3]

    pong_learner = DeepQLearner(net, optimizer, env, action_space)
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
train(args)