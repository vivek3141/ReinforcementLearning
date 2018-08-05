import gym
import numpy as np

env = gym.make("Taxi-v2")
Q = np.zeros([env.observation_space.n, env.action_space.n])
G = 0
a = 0.618
r = None
i = 0
for i in range(1, 2000):
    done = False
    G, r = 0, 0
    s = env.reset()
    while not done:
        action = np.argmax(Q[s])
        state, r, done, info = env.step(action=action)
        Q[s, action] = a * (r + np.max(Q[state]))
        G += r
        s = state
s = env.reset()
done = False
while not done:
    env.render()
    state, r, done, info = env.step(np.argmax(Q[s]))
    s = state
