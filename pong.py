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
import baselines.common.atari_wrappers as atari_wrappers


parser = argparse.ArgumentParser(description='Train a neural net to play pong.')
parser.add_argument('--net_file', metavar='net_file', type=str, nargs=1, default="",
                    help='file to load the model to be trained')
parser.add_argument('--render', default=False, action='store_true',
                    help='Render a screen to see the net play')
parser.add_argument('--headless', default=False, action='store_true',
                    help='Will save graphs to some path instead of displaying them')

M = 4 # take M steps and stack the resulting frames to produce the state
D = 10000 # memory buffer size
minibatch_size = 64 
discount = 0.99
start_learning_iteration = 1000 # start backprop after this many iterations
epoch_size = D/minibatch_size # after this many iterations, one epoch is complete
target_network_update_frequency = 1000 # after learning begins, update the target network after this many iterations
epsilons = np.linspace(1,0.02,100000) # epsilon annealment. uses the last value after they are all used.


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


def sample_minibatch(experiences, n):
    minibatch = []
    for i in range(0, n):
        minibatch.append(experiences[np.random.randint(0, len(experiences))])
    return minibatch


def print_memory_usage(device):
    process = psutil.Process(os.getpid())
    print("Memory usage (mb): {0}".format(process.memory_info().rss/1e6))
    if device.type == 'cuda':
        pass
        # Found that GPU usage is largely negligible (< 200mb)
        #print("GPU Memory Usage (mb): {0}".format(torch.cuda.memory_allocated(device)/1e6))


def create_target(minibatch, targetAgent):
    y = []
    for i in range(0, len(minibatch)):
        if minibatch[i].reward != 0: # for pong, for training purposes, we defined episode termination when there is a reward
            y.append(minibatch[i].reward)
        else:
            # technically reward = 0 here but wrote out the full expression as in the paper.
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


def save_net(net, optimizer):
    torch.save({
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, './net-' + random_string())


def get_sigint_handler(net, optimizer):
    def sigint_handler(signal_received, frame):
        print("saving net...")
        save_net(net, optimizer)
        sys.exit(0)
    return sigint_handler


def train(args):
    net = args['net']
    optmizer = args['optimizer']
    minibatch_size = args['minibatch_size']
    target_network_update_frequency = args['target_network_update_frequency']
    render = args['render']
    headless = args['headless']

    experiences = []
    env = gym.make('PongNoFrameskip-v4')
    env = atari_wrappers.FireResetEnv(env)
    env = atari_wrappers.NoopResetEnv(env)
    iteration = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = PongAgent(net, device)
    target_net = Net(M)
    deep_copy_nets(target_net, net)
    targetAgent = PongAgent(target_net, device)
    epsilon = None
    signal(SIGINT, get_sigint_handler(net, optimizer))
    episode_rewards = []
    while True: # iterate over multiple episodes until user exits
        env.reset()
        episode_reward = 0
        # create the initial observation
        lastMFrames = []
        for i in range(0, M):
            new_frame, _, _, _ = env.step(0) # stay still
            new_frame = preprocess_frame(new_frame)
            lastMFrames.append(new_frame)
        observation = torch.from_numpy(np.array(lastMFrames)).view(M, 159, 160).unsqueeze(0)
        done = False # episode not yet complete
        while not done: # iterate through episode
            print("Iteration {}:".format(iteration))
            print_memory_usage(device)
            # use the current observation to selection an action, run the environment, store the experience
            if render: # if we are rendering, mostly choose the optimal action
                epsilon = 0.05
            elif iteration < start_learning_iteration: # before learning, use first epsilon value
                epsilon = epsilons[0]
            elif iteration-start_learning_iteration < len(epsilons): # annealment
                epsilon = epsilons[iteration-start_learning_iteration]
            else: # after running out of epsilons, continue to use the last one
                epsilon = epsilons[-1]
            action = agent.epsilonGreedAction(observation, epsilon=epsilon)
            for i in range(0, M):
                new_frame, reward, done, info = env.step(action)
                if done: # if game is over, frame does not matter
                    break
                new_frame = preprocess_frame(new_frame)
                lastMFrames.pop(0)
                lastMFrames.append(new_frame)
                if reward != 0: # reward means a point was scored. Want that always to be the last frame.
                    episode_reward += reward
                    break
            if done: # done occurs right after the last reward. It is an empty frame. No need to add it to memory.
                break
            result_observation = torch.from_numpy(np.array(lastMFrames)).view(M, 159, 160).unsqueeze(0)
            experience = Experience(observation, action, result_observation, reward)
            if len(experiences) > D:
                experiences.pop(0)
            experiences.append(experience)
            observation = result_observation
            if render:
                env.render()
                sleep(0.1)
            # once enough experiences collected, start learning
            if iteration > start_learning_iteration: # only start learning after a number of experiences have been collected
                minibatch = sample_minibatch(experiences, minibatch_size)
                targets = create_target(minibatch, targetAgent)
                optimizer.zero_grad()   # zero the gradient buffers
                predictedStateActionValues = predictStateActionValues(minibatch, agent, device)
                loss = torch.sum((targets - predictedStateActionValues)**2)
                loss.backward()
                print("Loss: {}".format(loss))
                optimizer.step()
                if (iteration-start_learning_iteration) % target_network_update_frequency == 0:
                    deep_copy_nets(target_net, net)
            iteration = iteration + 1
        episode_rewards.append(episode_reward)
        plt.figure(0)
        plt.title('Reward v.s. Episode #')
        plt.plot(episode_rewards)
        plt.show(block=False)
        plt.pause(0.1)

    env.close()

args = parser.parse_args()

net_file = args.net_file[0] if len(args.net_file) == 1 else ''
net = Net(M)
optimizer = optim.SGD(net.parameters(), lr=0.005)
if net_file != '':
    checkpoint = torch.load(net_file)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

args = {
    'net': net,
    'optimizer': optimizer,
    'minibatch_size': minibatch_size,
    'target_network_update_frequency': target_network_update_frequency,
    'render': args.render,
    'headless': args.headless
}
train(args)