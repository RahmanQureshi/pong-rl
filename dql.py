
import torch
import random
import string
import numpy as np
import psutil
import os
import copy
from time import sleep
import matplotlib.pyplot as plt
import torch.optim as optim
from circle_buffer import CircleBuffer
from experience import Experience
from model import Net


class DeepQLearner:


    def __init__(self, env, action_space, net=None, optimizer=None, checkpoint='', replay_buffer_size=10000, render=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = Net(2)
        self.optimizer = optim.RMSprop(self.net.parameters(), lr=0.0025, alpha=0.9, eps=1e-02, momentum=0.0)
        # if checkpoint is provided, overwrite the state dictionaries of the net and optimizer
        if checkpoint != '':
            self.load(checkpoint)
        self.target_net = copy.deepcopy(self.net)
        self.net.to(self.device)
        self.target_net.to(self.device)
        self.replay_buffer = CircleBuffer(replay_buffer_size)
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
        self.render = render
        # dummy input is used for convenience when creating target values for a training batch
        self.dummy_input = self.env.reset()
        # used for saving stats and checkpoint
        self.rand_string = self.random_string()


    def learn(self, minibatch_size=32):
        """DeepQLearner will look over all its experiences and do a single learning iteration step.
        """
        minibatch = self.replay_buffer.sample(minibatch_size)
        # construct the target values
        batch_result_states = torch.stack([e.result_state for e in minibatch])
        max_target_Q_values = self.Q(self.target_net, batch_result_states).max(dim=1)
        targets = torch.tensor([e.reward if e.terminal else e.reward + self.discount*max_target_Q_values[0][i].item() for i,e in enumerate(minibatch)])
        # compute the predicted Q values of all the experiences (state and actions taken) in the batch
        batch_states = torch.stack([e.state for e in minibatch]) 
        predicted_Q_values = torch.empty(minibatch_size)
        for i,q in enumerate(self.Q(self.net, batch_states)):
            predicted_Q_values[i] = q[self.action_space.index(minibatch[i].action)] 
        # backward propagation
        self.optimizer.zero_grad()
        loss = self.lossfct(predicted_Q_values, targets)
        print("Loss: {}".format(loss))
        loss.backward()
        self.optimizer.step()
        self.num_learning_steps = self.num_learning_steps + 1
        if self.num_learning_steps % self.target_network_update_frequency == 0:
            self.deep_copy_nets()


    def train(self, num_episodes):
        for i in range(0, num_episodes):
            observation = self.env.reset()
            done = False
            episode_reward = 0
            while not done:
                action = self.epsilon_greedy_action(observation)
                result_observation, reward, done, info = self.env.step(action)
                # To plot while debugging: plt.imshow(result_observation.numpy().squeeze(0))
                if self.render:
                    self.env.render()
                    sleep(0.1)
                terminal = done or 'done' in info
                self.replay_buffer.push(Experience(observation, action, result_observation, reward, terminal))
                observation = result_observation
                self.num_env_steps = self.num_env_steps + 1
                print("NumEnvSteps: {}".format(self.num_env_steps))
                episode_reward = episode_reward + reward
                if self.num_env_steps > self.start_learning_iteration:
                    self.learn()
                self.print_memory_usage(self.device)
            # Episode is done. Save current status and plots.
            self.episode_rewards.append(episode_reward)
            if not render:
                self.save_plots()
                self.save(self.net, self.optimizer)


    def Q(self, net, x):
        """ Wrapper around NN forward which turns the tensor to a float type and moves it onto the device.
            Returns the value of all actions given the state.
        """
        return net(x.float().to(self.device))


    def get_epsilon(self):
        if self.render:
            return 0
        if self.num_env_steps < self.start_learning_iteration: # before learning, use first epsilon value
            return self.epsilons[0]
        elif self.num_env_steps-self.start_learning_iteration < len(self.epsilons): # annealment
            return self.epsilons[self.num_env_steps-self.start_learning_iteration]
        # after running out of epsilons, continue to use the last one
        return self.epsilons[-1]


    def epsilon_greedy_action(self, x):
        """Accepts a single state and returns an action based on
           epsilon greedy exploration v.s. exploitation strategy.
        """
        epsilon = self.get_epsilon()
        if np.random.uniform(0,1) <= epsilon: # with probability epsilon, pick random action
            return np.random.choice(self.action_space)
        x = x.unsqueeze(0) # add batch size dimension
        return self.action_space[self.Q(self.net, x).argmax().item()]


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
        plt.savefig('reward_vs_episode_{}.png'.format(self.rand_string))
    

    def print_memory_usage(self, device):
        process = psutil.Process(os.getpid())
        print("Memory usage (mb): {0}".format(process.memory_info().rss/1e6))
        if device.type == 'cuda':
            pass
            # Found that GPU usage is largely negligible (< 200mb) for 32 size batch
            #print("GPU Memory Usage (mb): {0}".format(torch.cuda.memory_allocated(device)/1e6))


    def save(self, net, optimizer):
        torch.save({
                    'model_state_dict': self.net.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()
                    }, './dql_checkpoint_' + self.rand_string)

    def load(self, file):
        checkpoint = torch.load(file)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    def random_string(self, string_length=10):
        """Generate_sa random string of fixed length """
        lettersAndNumbers = string.ascii_lowercase + string.digits
        return ''.join(random.choice(lettersAndNumbers) for i in range(string_length))