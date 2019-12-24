import gym
from time import sleep
from pong_agent import PongAgent, Net
import torch

env = gym.make('Pong-v0')

net = Net()
agent = PongAgent(net)

for i_episode in range(100):
    observation = env.reset()
    for t in range(100):
        env.render()
        action = agent.getAction(observation)
        observation, reward, done, info = env.step(action)
        sleep(0.1)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
