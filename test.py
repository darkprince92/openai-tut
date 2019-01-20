import gym
import pprint as pp

env = gym.make('CartPole-v0')

for i_episode in range(10):
    observation = env.reset()
    print(observation)

    for t in range(200):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        pp.pprint("{}, {}, {}, {}".format(t, observation, reward, done))
        if done:
            print("Episode {} finished after {} time steps".format(i_episode, t + 1))
            break

env.close()
