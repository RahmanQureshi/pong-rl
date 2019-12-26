import gym
from time import sleep
from pong_agent import PongAgent, Net
import torch
import numpy as np
import torch.optim as optim
import os
import psutil
import matplotlib.pyplot as plt

D = 10000 # memory buffer size
minibatch_size = 32
discount = 0.99
start_learning_iteration = D # start backprop after this many iterations
plot_every_num_iterations = int(D/minibatch_size) # after this many iterations, one epoch is complete

class Experience:

    def __init__(self, state, action, result_state, reward, done):
        self.state = torch.from_numpy(state).view(3, 210, 160).unsqueeze(0)
        self.action = action
        self.result_state = torch.from_numpy(result_state).view(3, 210, 160).unsqueeze(0)
        self.reward = reward
        self.done = done


def sample_minibatch(experiences, n):
    minibatch = []
    for i in range(0, n):
        minibatch.append(experiences[np.random.randint(0, len(experiences))])
    return minibatch


def print_memory_usage():
    process = psutil.Process(os.getpid())
    print("Memory usage (mb): {0}".format(process.memory_info().rss/1e6))


def create_target(minibatch, target_net):
    y = []
    targetAgent = PongAgent(target_net)
    for i in range(0, len(minibatch)):
        if minibatch[i].done:
            y.append(minibatch[i].reward)
        else:
            y.append(minibatch[i].reward + discount*targetAgent.getOptimalActionValue(minibatch[i].result_state))
    return torch.tensor(y)


def predictOptimalActionValues(minibatch, net):
    prediction = torch.empty(len(minibatch))
    batch_states = minibatch[0].state
    for i in range(1, len(minibatch)):
        batch_states = torch.cat((batch_states, minibatch[i].state))
    net_output = net(batch_states.float())
    for i in range(0, len(minibatch)):
        prediction[i] = net_output[i][minibatch[i].action]
    return prediction


def deep_copy_nets(target_net, net):
    """Copies parameters of net into target_net
    """
    params1 = net.named_parameters()
    params2 = target_net.named_parameters()
    dict_params2 = dict(params2)
    for name1, param1 in params1:
        dict_params2[name1].data.copy_(param1.data)


def train(target_net, net, num_episodes=10, minibatch_size=32, target_network_update_frequency=10000):
    experiences = []
    env = gym.make('Pong-v0')
    agent = PongAgent(net)
    optimizer = optim.SGD(net.parameters(), lr=0.001)
    j = 0
    losses = []
    avg_optimal_action_values = []
    iteration = 0
    for i in range(0, num_episodes):
        observation = env.reset()
        done = False
        print("Episode: {}", i)
        while not done:
            print("Iteration {}:".format(iteration))
            print_memory_usage()
            epsilon = 1
            action = agent.epsilonGreedAction(observation, epsilon)
            result_observation, reward, done, info = env.step(action)
            experience = Experience(observation, action, result_observation, reward, done)
            if len(experiences) > D:
                experiences.pop(0)
            experiences.append(experience)
            if iteration > start_learning_iteration: # only start learning after a number of experiences have been collected
                minibatch = sample_minibatch(experiences, minibatch_size)
                targets = create_target(minibatch, target_net)
                optimizer.zero_grad()   # zero the gradient buffers
                predictedOptimalActionValues = predictOptimalActionValues(minibatch, net)
                loss = torch.sum((targets - predictedOptimalActionValues)**2)
                print("Loss: {}".format(loss))
                loss.backward()
                optimizer.step()
                if j == target_network_update_frequency:
                    j = 0
                    deep_copy_nets(target_net, net)
                j = j + 1
                if (iteration-start_learning_iteration) % plot_every_num_iterations == 0:
                    losses.append(loss.item())
                    avg_optimal_action_values.append(predictedOptimalActionValues.mean().item())
                    plt.figure(0)
                    plt.title("loss")
                    plt.plot(losses)
                    plt.figure(1)
                    plt.title("average optimal action value")
                    plt.plot(avg_optimal_action_values)
                    plt.show(block=False)
                    plt.pause(1)
            iteration = iteration + 1
    env.close()


net = Net()
target_net = net
train(target_net, net, num_episodes=100000, minibatch_size=minibatch_size)