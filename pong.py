"""Trains a neural net to play Pong.
   Exit using Ctrl-C. This will automatically save the neural network into file net-<random string>.
"""
import gym
from time import sleep
from pong_agent import PongAgent, Net
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


parser = argparse.ArgumentParser(description='Train a neural net to play pong.')
parser.add_argument('--net_file', metavar='net_file', type=str, nargs=1, default="",
                    help='file to load the model to be trained')

M = 2 # run the game M times and stack the resulting frames to produce the state
D = 10000 # memory buffer size
minibatch_size = 32
discount = 0.99
start_learning_iteration = D # start backprop after this many iterations
plot_every_num_iterations = int(D/minibatch_size) # after this many iterations, one epoch is complete
target_network_update_frequency = 10000 # after learning begins, update the target network after this many iterations
epsilons = np.linspace(1,0.05,100000) # epsilon annealment. uses the last value after they are all used.


class Experience:


    def __init__(self, state, action, result_state, reward, done):
        self.state = state
        self.action = action
        self.result_state = result_state
        self.reward = reward
        self.done = done


def rgb_frame_to_grayscale(frame):
    """Convert rgb to grayscale. Uses observed luminance to compute grey value.
    Input:
        numpy array 210x160x3
    Returns:
        numpy array 210x160
    """
    rgb = np.array([0.299, 0.587, 0.114])
    return np.inner(rgb, frame).astype(np.uint8)


def sample_minibatch(experiences, n):
    minibatch = []
    for i in range(0, n):
        minibatch.append(experiences[np.random.randint(0, len(experiences))])
    return minibatch


def print_memory_usage(device):
    process = psutil.Process(os.getpid())
    print("Memory usage (mb): {0}".format(process.memory_info().rss/1e6))
    if device.type == 'cuda':
        print("GPU Memory Usage: {0}".format(torch.cuda.memory_allocated(device)/1e6))


def create_target(minibatch, targetAgent):
    y = []
    for i in range(0, len(minibatch)):
        if minibatch[i].done:
            y.append(minibatch[i].reward)
        else:
            y.append(minibatch[i].reward + discount*targetAgent.getOptimalActionValue(minibatch[i].result_state))
    return torch.tensor(y)


def predictStateActionValues(minibatch, agent, device):
    """For each experience in the minibatch, compute Q(s,a).
    """
    prediction = torch.empty(len(minibatch))
    batch_states = minibatch[0].state
    for i in range(1, len(minibatch)):
        batch_states = torch.cat((batch_states, minibatch[i].state))
    net_output = agent.batchPredict(batch_states)
    for i in range(0, len(minibatch)):
        prediction[i] = net_output[i][minibatch[i].action]
    return prediction


def random_string(stringLength=10):
    """Generate_sa random string of fixed length """
    lettersAndNumbers = string.ascii_lowercase + string.digits
    return ''.join(random.choice(lettersAndNumbers) for i in range(stringLength))


def deep_copy_nets(target_net, net):
    """Copies parameters of net into target_net
    """
    params1 = net.named_parameters()
    params2 = target_net.named_parameters()
    dict_params2 = dict(params2)
    for name1, param1 in params1:
        dict_params2[name1].data.copy_(param1.data)


def save_net(net):
    torch.save(net.state_dict(), "./net-" + random_string())


def get_sigint_handler(net):
    def sigint_handler(signal_received, frame):
        print("saving net...")
        save_net(net)
        sys.exit(0)
    return sigint_handler


def train(net, minibatch_size=32, target_network_update_frequency=10000):
    experiences = []
    env = gym.make('Pong-v0')
    optimizer = optim.SGD(net.parameters(), lr=0.001)
    losses = []
    avg_action_values = []
    iteration = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = PongAgent(net, device)
    target_net = Net()
    deep_copy_nets(target_net, net)
    targetAgent = PongAgent(target_net, device)
    lastMFrames = []
    epsilon = None
    while True:
        init_frame = rgb_frame_to_grayscale(env.reset())
        observation = np.array([init_frame, init_frame])
        observation = torch.from_numpy(observation).view(2, 210, 160).unsqueeze(0)
        done = False
        while not done:
            print("Iteration {}:".format(iteration))
            print_memory_usage(device)
            # use the current observation to selection an action, run the environment, store the experience
            if iteration <= start_learning_iteration: # before learning, use first epsilon value
                epsilon = epsilons[0]
            elif iteration-start_learning_iteration < len(epsilons): # annealment
                epsilon = epsilons[iteration-start_learning_iteration]
            else: # after running out of epsilons, continue to use the last one
                epsilon = epsilons[-1]
            action = agent.epsilonGreedAction(observation, epsilon=epsilon)
            result_observation = []
            for i in range(0, M):
                new_frame, reward, done, info = env.step(action)
                new_frame = rgb_frame_to_grayscale(new_frame)
                result_observation.append(new_frame)
            result_observation = np.array(result_observation)
            result_observation = torch.from_numpy(result_observation).view(2, 210, 160).unsqueeze(0)
            experience = Experience(observation, action, result_observation, reward, done)
            if len(experiences) > D:
                experiences.pop(0)
            experiences.append(experience)
            observation = result_observation
            # once enough experiences collected, start learning
            if iteration > start_learning_iteration: # only start learning after a number of experiences have been collected
                minibatch = sample_minibatch(experiences, minibatch_size)
                targets = create_target(minibatch, targetAgent)
                optimizer.zero_grad()   # zero the gradient buffers
                predictedStateActionValues = predictStateActionValues(minibatch, agent, device)
                loss = torch.sum((targets - predictedStateActionValues)**2)
                print("Loss: {}".format(loss))
                loss.backward()
                optimizer.step()
                if iteration % target_network_update_frequency == 0:
                    deep_copy_nets(target_net, net)
                if iteration % plot_every_num_iterations == 0:
                    losses.append(loss.item())
                    avg_action_values.append(predictedStateActionValues.mean().item())
                    plt.figure(0)
                    plt.title("loss")
                    plt.plot(losses)
                    plt.figure(1)
                    plt.title("average optimal action value")
                    plt.plot(avg_action_values)
                    plt.show(block=False)
                    plt.pause(1)
            iteration = iteration + 1
    env.close()

args = parser.parse_args()
net_file = args.net_file[0] if len(args.net_file) == 1 else ''
net = Net()
if net_file != '':
    net = torch.load(net_file)
signal(SIGINT, get_sigint_handler(net))
train(net, minibatch_size=minibatch_size, target_network_update_frequency=target_network_update_frequency)
