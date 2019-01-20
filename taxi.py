import gym
import numpy as np
import random
import pprint as pp

env = gym.make('Taxi-v2')
print(env.action_space)
print(env.observation_space)
print(env.reward_range)

observation = env.reset()
env.render()
print(observation)

q_table = np.zeros([env.observation_space.n, env.action_space.n])

alpha = 0.1
gamma = 0.6
epsilon = 0.1

all_epochs = []
all_penalties = []

MAX_EPISODES = 1000
PENALTY = -10

for i in range(MAX_EPISODES):
    observation = env.reset()

    epochs, penalties, rewards = 0, 0, 0
    done = False

    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[observation])

        next_observation, reward, done, info = env.step(action)

        old_value = q_table[observation, action]
        next_max = np.max(q_table[next_observation])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[observation, action] = new_value

        if reward == PENALTY:
            penalties += 1

        observation = next_observation
        epochs += 1

    if i%100 == 0:
        print("Episode: {}".format(i))

print("Finished")
