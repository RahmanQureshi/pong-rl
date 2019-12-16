import gym
from time import sleep
from pong_agent import PongAgent

env = gym.make('Pong-v0')

agent = PongAgent()
print(env.observation_space)
exit()

for i_episode in range(100):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = agent.getAction(observation)
        observation, reward, done, info = env.step(action)
        sleep(0.1)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
